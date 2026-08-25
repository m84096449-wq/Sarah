import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, theta=10000.0):
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        positions = torch.arange(max_seq_len).float()

        freqs = torch.outer(positions, inv_freq)

        self.register_buffer(
            "cos",
            freqs.cos(),
            persistent=False
        )

        self.register_buffer(
            "sin",
            freqs.sin(),
            persistent=False
        )

    def forward(self, q, k):
        seq_len = q.shape[-2]

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        def rotate(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]

            rotated = torch.stack(
                (-x2, x1),
                dim=-1
            )

            return rotated.flatten(-2)

        q = q * cos.repeat_interleave(2, dim=-1)
        q = q + rotate(q) * sin.repeat_interleave(2, dim=-1)

        k = k * cos.repeat_interleave(2, dim=-1)
        k = k + rotate(k) * sin.repeat_interleave(2, dim=-1)

        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.hidden_size % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta
        )

    def forward(self, x):
        batch, seq_len, hidden = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        q, k = self.rope(q, k)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            causal_mask,
            torch.finfo(scores.dtype).min
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention,
            v
        )

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch,
            seq_len,
            hidden
        )

        return self.o_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False
        )

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x))
            * self.up_proj(x)
        )


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(
            config.hidden_size,
            config.norm_eps
        )

        self.attention = CausalSelfAttention(config)

        self.ffn_norm = RMSNorm(
            config.hidden_size,
            config.norm_eps
        )

        self.ffn = SwiGLU(config)

    def forward(self, x):
        x = x + self.attention(
            self.attn_norm(x)
        )

        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x


class SarahModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.norm_eps
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits

    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
        )
