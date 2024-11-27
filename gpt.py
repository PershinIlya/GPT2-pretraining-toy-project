from dataclasses import dataclass 
import os
import math
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import inspect
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0   
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12  # original GPT-2 has 12 layers
    n_head: int = 12  # original GPT-2 has 12 heads
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T) where T is the sequence length
        B, T = idx.size()
        assert T <= self.config.block_size, f'Cannot forward, model block size is exhausted, {T} > {self.config.block_size}'
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)    
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        # calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def save_checkpoint(self, model, optimizer, step, loss, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"checkpoint_{timestamp}_step_{step}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")



    # @classmethod
    # def from_pretrained(cls, model_type):
    #     """Loads pretrained GPT-2 model weights from huggingface"""
    #     assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    #     from transformers import GPT2LMHeadModel
    #     print("loading weights from pretrained gpt: %s" % model_type)

    #     # n_layer, n_head and n_embd are determined from model_type
    #     config_args = {
    #         'gpt2':      dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    #         'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    #         'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    #         'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    #     }[model_type]
    #     config_args['vocab_size'] = 50257
    #     config_args['block_size'] = 1024
    #     config = GPTConfig(**config_args)
    #     model = GPT(config)
    #     sd = model.state_dict()
    #     sd_keys = sd.keys()
    #     sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()

    #     # copy while ensuring all the parameters are aligned and match in names and shapes
    #     sd_keys_hf = sd_hf.keys()
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
    #     transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    #     assert len(sd_keys) == len(sd_keys_hf), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
    #     for k in sd_keys_hf:
    #         if any(k.endswith(t) for t in transposed):
    #             assert sd_hf[k].shape[::-1] == sd[k].shape 
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k].t())
    #         else:
    #             assert sd_hf[k].shape == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k])

    #     return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim gropus. Any parameters that is 2D will be weight decayed, otherwise no. 
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't. 
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tenssors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tenssors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
        # create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, master_process, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ('train', 'val')

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # move the position in the tensor
        self.current_position += B * T * self.num_processes
        # # if loading the next batch would overrun the tokens, reset the position
        # if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
        #     self.current_position = self.B * self.T * self.process_rank

        # if loading the next batch would be out of bounds, advance to the next shard
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y