import os
import math
import time
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from hellaswag import render_example, iterate_examples
import tiktoken
import numpy as np

# -------------------- Model Definitions --------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        assert seq_len <= self.config.block_size, f"Sequence length {seq_len} exceeds block size {self.config.block_size}"
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=idx.device)
        pos_embeddings = self.transformer.wpe(pos_ids)
        token_embeddings = self.transformer.wte(idx)
        x = token_embeddings + pos_embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name):
        assert model_name in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading pretrained GPT weights: {model_name}")
        config_map = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }
        config_args = config_map[model_name]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        state_dict = model.state_dict()
        state_dict_keys = [k for k in state_dict.keys() if not k.endswith('.attn.bias')]
        pretrained_model = GPT2LMHeadModel.from_pretrained(model_name)
        pretrained_state_dict = pretrained_model.state_dict()
        pretrained_keys = [k for k in pretrained_state_dict.keys() if not (k.endswith('.attn.masked_bias') or k.endswith('.attn.bias'))]
        weights_to_transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(pretrained_keys) == len(state_dict_keys), f"Mismatch in state dict keys count"
        for key in pretrained_keys:
            if any(key.endswith(w) for w in weights_to_transpose):
                assert pretrained_state_dict[key].shape[::-1] == state_dict[key].shape
                with torch.no_grad():
                    state_dict[key].copy_(pretrained_state_dict[key].t())
            else:
                assert pretrained_state_dict[key].shape == state_dict[key].shape
                with torch.no_grad():
                    state_dict[key].copy_(pretrained_state_dict[key])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        if master_process:
            print(f"Weight decay params: {len(decay_params)} tensors, {sum(p.numel() for p in decay_params):,} params")
            print(f"No decay params: {len(no_decay_params)} tensors, {sum(p.numel() for p in no_decay_params):,} params")
        use_fused_adamw = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        if master_process:
            print(f"Using fused AdamW optimizer: {use_fused_adamw}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused_adamw)
        return optimizer

# -------------------- Data Loading --------------------

def load_token_tensor(filename):
    numpy_array = np.load(filename).astype(np.int32)
    tensor = torch.tensor(numpy_array, dtype=torch.long)
    return tensor

class DataLoaderLite:
    def __init__(self, batch_size, seq_len, rank, world_size, split):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        assert split in {'train', 'val'}
        data_root = "edu_fineweb10B"
        shards = sorted([os.path.join(data_root, f) for f in os.listdir(data_root) if split in f])
        assert shards, f"No shards found for split '{split}'"
        self.shards = shards
        if master_process:
            print(f"Found {len(shards)} shards for split '{split}'")
        self.reset()

    def reset(self):
        self.current_shard_idx = 0
        self.tokens = load_token_tensor(self.shards[self.current_shard_idx])
        self.position = self.batch_size * self.seq_len * self.rank

    def next_batch(self):
        b, t = self.batch_size, self.seq_len
        buffer = self.tokens[self.position:self.position + b * t + 1]
        x = buffer[:-1].view(b, t)
        y = buffer[1:].view(b, t)
        self.position += b * t * self.world_size
        if self.position + (b * t * self.world_size + 1) > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.tokens = load_token_tensor(self.shards[self.current_shard_idx])
            self.position = b * t * self.rank
        return x, y

# -------------------- Evaluation Helpers --------------------

def get_most_likely_completion(tokens, mask, logits):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_tokens = tokens[..., 1:].contiguous()
    flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flat_tokens = shifted_tokens.view(-1)
    losses = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
    losses = losses.view(tokens.size(0), -1)
    shifted_mask = mask[..., 1:].contiguous()
    masked_losses = losses * shifted_mask
    sum_loss = masked_losses.sum(dim=1)
    avg_loss = sum_loss / shifted_mask.sum(dim=1)
    predicted_idx = avg_loss.argmin().item()
    return predicted_idx

# -------------------- Distributed Setup --------------------

ddp_enabled = int(os.environ.get('RANK', -1)) != -1
if ddp_enabled:
    assert torch.cuda.is_available(), "DDP requires CUDA"
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
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

tokenizer = tiktoken.get_encoding("gpt2")

total_tokens_per_batch = 524288
batch_size = 64
sequence_length = 1024
assert total_tokens_per_batch % (batch_size * sequence_length * ddp_world_size) == 0
grad_accumulation_steps = total_tokens_per_batch // (batch_size * sequence_length * ddp_world_size)

if master_process:
    print(f"Total batch size: {total_tokens_per_batch}")
    print(f"Gradient accumulation steps: {grad_accumulation_steps}")

train_data_loader = DataLoaderLite(batch_size, sequence_length, ddp_rank, ddp_world_size, "train")
val_data_loader = DataLoaderLite(batch_size, sequence_length, ddp_rank, ddp_world_size, "val")

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

use_torch_compile = False
if use_torch_compile:
    model = torch.compile(model)
if ddp_enabled:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp_enabled else model

max_learning_rate = 6e-4
min_learning_rate = max_learning_rate * 0.1
warmup_iterations = 715
max_iterations = 19073

def get_learning_rate(step):
    if step < warmup_iterations:
        return max_learning_rate * (step + 1) / warmup_iterations
    if step > max_iterations:
        return min_learning_rate
    decay_ratio = (step - warmup_iterations) / (max_iterations - warmup_iterations)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_learning_rate, device_type=device_type)

log_directory = "log"
os.makedirs(log_directory, exist_ok=True)
log_path = os.path.join(log_directory, "log.txt")
with open(log_path, "w") as f:
    pass  # clear the log file

for step in range(max_iterations):
    start_time = time.time()
    is_last_step = (step == max_iterations - 1)

    if step % 250 == 0 or is_last_step:
        model.eval()
        val_data_loader.reset()
        val_loss_sum = 0.0
        val_loss_batches = 20
        with torch.no_grad():
            for _ in range(val_loss_batches):
                inputs, targets = val_data_loader.next_batch()
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(inputs, targets)
                val_loss_sum += (loss / val_loss_batches).detach()
        if ddp_enabled:
            dist.all_reduce(val_loss_sum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_sum.item():.4f}")
            with open(log_path, "a") as f:
                f.write(f"{step} val {val_loss_sum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or is_last_step):
                checkpoint_path = os.path.join(log_directory, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_sum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    if (step % 250 == 0 or is_last_step) and not use_torch_compile:
        correct_predictions_norm = 0
        total_examples = 0
        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(tokens)
                predicted_norm = get_most_likely_completion(tokens, mask, logits)
            total_examples += 1
            correct_predictions_norm += int(predicted_norm == label)
        if ddp_enabled:
            total_examples_tensor = torch.tensor(total_examples, dtype=torch.long, device=device)
            correct_predictions_tensor = torch.tensor(correct_predictions_norm, dtype=torch.long, device=device)
            dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_predictions_tensor, op=dist.ReduceOp.SUM)
            total_examples = total_examples_tensor.item()
            correct_predictions_norm = correct_predictions_tensor.item()
        accuracy_norm = correct_predictions_norm / total_examples
        if master_process:
            print(f"HellaSwag accuracy: {correct_predictions_norm}/{total_examples} = {accuracy_norm:.4f}")
            with open(log_path, "a") as f:
                f.write(f"{step} hella {accuracy_norm:.4f}\n")

    if ((step > 0 and step % 250 == 0) or is_last_step) and not use_torch_compile:
        model.eval()
        num_samples = 4
        max_gen_length = 32
        prompt_tokens = tokenizer.encode("Hello, I'm a language model,")
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).repeat(num_samples, 1).to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        generated = prompt_tensor
        while generated.size(1) < max_gen_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(generated)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                chosen_idx = torch.multinomial(topk_probs, 1, generator=sample_rng)
                next_tokens = torch.gather(topk_indices, -1, chosen_idx)
                generated = torch.cat([generated, next_tokens], dim=1)
        for sample_idx in range(num_samples):
            sample_tokens = generated[sample_idx, :max_gen_length].tolist()
            sample_text = tokenizer.decode(sample_tokens)
            print(f"Rank {ddp_rank} sample {sample_idx}: {sample_text}")

    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    for micro_step in range(grad_accumulation_steps):
        inputs, targets = train_data_loader.next_batch()
        inputs, targets = inputs.to(device), targets.to(device)
        if ddp_enabled:
            model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(inputs, targets)
        loss = loss / grad_accumulation_steps
        accumulated_loss += loss.detach()
        loss.backward()
    if ddp_enabled:
        dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    current_lr = get_learning_rate(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    tokens_processed = train_data_loader.batch_size * train_data_loader.seq_len * grad_accumulation_steps * ddp_world_size
    tokens_per_second = tokens_processed / elapsed_time
    if master_process:
        print(f"Step {step:5d} | Loss: {accumulated_loss.item():.6f} | LR: {current_lr:.4e} | Grad Norm: {grad_norm:.4f} | Time: {elapsed_time*1000:.2f}ms | Tokens/sec: {tokens_per_second:.2f}")
        with open(log_path, "a") as f:
            f.write(f"{step} train {accumulated_loss.item():.6f}\n")

if ddp_enabled:
    destroy_process_group()
