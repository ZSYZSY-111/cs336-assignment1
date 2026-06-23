"""Transformer LM with Grouped-Query Attention (GQA).

Reuses all building blocks from train_transformer.py; only the attention
module is replaced by GroupedQueryAttention. Set num_kv_heads < num_heads
to enable GQA (num_kv_heads == num_heads degrades to standard MHA).
"""
import torch
import torch.nn as nn

from train_transformer import (
    Embedding,
    Linear,
    RMSNorm,
    ROPE,
    SwiGLU,
    scaled_dot_product_attention,
)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,
        device=None,
        dtype=None,
        use_ROPE=False,
        theta=None,
        max_seq_len=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads  # how many Q heads share one KV head

        # Q stays full size; K/V are smaller (num_kv_heads instead of num_heads)
        self.q_proj = Linear(d_model, num_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.use_ROPE = use_ROPE
        if use_ROPE:
            self.rope = ROPE(theta, self.head_dim, max_seq_len, device=device)

    def forward(self, x, token_positions=None):
        batch, seq, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape into heads
        Q = Q.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_ROPE:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device).unsqueeze(0)
            token_positions = token_positions.unsqueeze(1)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # expand KV heads to match Q heads: each KV head is shared by num_groups Q heads
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        mask = torch.tril(torch.ones(seq, seq, device=x.device, dtype=torch.bool))

        score = scaled_dot_product_attention(Q, K, V, mask)
        score = score.transpose(1, 2).contiguous().reshape(batch, seq, self.num_heads * self.head_dim)

        return self.output_proj(score)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,
        d_ff,
        max_seq_len,
        theta,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = GroupedQueryAttention(
            d_model,
            num_heads,
            num_kv_heads,
            device=device,
            dtype=dtype,
            use_ROPE=True,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        normed = self.ln1(x)
        attn_out = self.attn(normed, token_positions=token_positions)
        x = x + attn_out

        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        num_kv_heads,
        d_ff,
        rope_theta,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                num_kv_heads,
                d_ff,
                context_length,
                rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices):
        batch, seq = in_indices.shape
        token_positions = torch.arange(seq, device=in_indices.device).unsqueeze(0)

        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
