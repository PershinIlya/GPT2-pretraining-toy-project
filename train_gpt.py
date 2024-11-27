import os
import math
import time
import torch
import tiktoken
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch._dynamo

from gpt import GPT, GPTConfig, DataLoaderLite
from hellaswag import render_example, iterate_examples, get_most_likely_row

torch._dynamo.config.cache_size_limit = 64

def main():
    from torch.distributed import init_process_group, destroy_process_group

    print(f"RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

    # Set up DDP
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), "CUDA is required for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    total_batch_size = 524288  # in number of tokens
    B = 16  # micro batch size
    T = 1024  # sequence length
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    print("I am GPU", ddp_rank)

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, master_process=master_process, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, master_process=master_process, split="val")

    torch.set_float32_matmul_precision('high')

    # Create the base model
    base_model = GPT(GPTConfig(vocab_size=50304))
    base_model.to(device)

    # Create the training model (compiled)
    training_model = base_model
    if ddp:
        training_model = DDP(training_model, device_ids=[ddp_local_rank])
    training_model = torch.compile(training_model)

    # Create the evaluation model (uncompiled, not wrapped with DDP)
    evaluation_model = GPT(GPTConfig(vocab_size=50304))
    evaluation_model.to(device)

    # Optimizer
    optimizer = training_model.module.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    # Rest of your code...
    max_lr = 6e-4 * 3
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073
    eval_interval = 100

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    if master_process:
        with open(log_file, "w") as f:
            pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Training loop
        training_model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = training_model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                training_model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(training_model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size)
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

        # Evaluation steps
        if step % eval_interval == 0 or last_step:
            # Before evaluation, load the state_dict from the training model to the evaluation model
            evaluation_model.load_state_dict(training_model.module.state_dict())
            
            if master_process:
                evaluation_model.save_checkpoint(evaluation_model, optimizer, step, loss_accum.item())

            # Validation loss evaluation
            evaluation_model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = evaluation_model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            # HellaSwag evaluation
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # Distribute examples across GPUs
                if i % ddp_world_size != ddp_rank:
                    continue
                # Render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # Get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = evaluation_model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # Reduce the stats across all processes
            if ddp:
                num_total_tensor = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm_tensor = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm_tensor, op=dist.ReduceOp.SUM)
                num_total = num_total_tensor.item()
                num_correct_norm = num_correct_norm_tensor.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

            # Generation (if needed)
            if (step > 0) and (step % eval_interval == 0) or last_step:
                evaluation_model.eval()
                num_return_sequences = 4
                max_length = 32
                tokens = enc.encode("Hello, I'm a language model,")
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                xgen = tokens.to(device)
                sample_rng = torch.Generator(device=device)
                sample_rng.manual_seed(42 + ddp_rank)
                while xgen.size(1) < max_length:
                    with torch.no_grad():
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, _ = evaluation_model(xgen)
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                        ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                        xcol = torch.gather(topk_indices, -1, ix)
                        xgen = torch.cat((xgen, xcol), dim=1)
                # Print generated text
                for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = enc.decode(tokens)
                    print(f"rank {ddp_rank} sample {i}: {decoded}")

    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()

# ddp launch:
# torchrun --standalone --nproc_per_node=4 train_gpt.py