import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        weight = torch.empty((out_features, in_features), device = device, dtype= dtype)
        self.weight = nn.Parameter(weight)

        torch.nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x):
        return x @ self.weight.T
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        weight = torch.empty((num_embeddings, embedding_dim), device= device, dtype= dtype)
        self.weight = nn.Parameter(weight)

        torch.nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = 1.0,
            a=-3,
            b=3,
        )

    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5, device = None, dtype = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device= device, dtype= dtype))

    def forward(self, x):
        mean = torch.mean(x ** 2, dim = -1, keepdim= True)
        rms = torch.sqrt(mean + self.eps)
        normalized = x/rms
        return normalized * self.weight


def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device = None, dtype = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device = device, dtype = dtype)
        self.w2 = Linear(d_ff, d_model, device = device, dtype = dtype)
        self.w3 = Linear(d_model, d_ff, device = device, dtype = dtype)

    def forward(self, x):
        w1 = self.w1(x)
        w3 = self.w3(x)

        w1_silu = silu(w1)
        w2 = self.w2(w1_silu * w3)
        return w2

def scaled_dot_product_attention(Q, K, V, mask = None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim = -1)
    out = attn @ V
    return out

class ROPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device = None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        i = torch.arange(0, d_k // 2, device = device)
        inv_freq = 1.0 / (theta ** (2 * i / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x, token_positions):
        assert x.shape[-1] == self.d_k
        angles = token_positions[..., None] * self.inv_freq
        cos = torch.cos(angles).to(dtype= x.dtype)
        sin = torch.sin(angles).to(dtype= x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        rotated = torch.stack((rotated_even, rotated_odd), dim = -1)
        return rotated.flatten(-2)

        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device = None, dtype = None, use_ROPE = False, theta = None, max_seq_len = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.k_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.v_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.output_proj = Linear(d_model, d_model, device = device, dtype = dtype)

        self.use_ROPE = use_ROPE
        if use_ROPE:
            self.rope = ROPE(theta, self.head_dim, max_seq_len, device = device)
    

    def forward(self, x, token_positions = None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        batch, seq, d_model = Q.shape

        Q = Q.reshape(batch, seq, self.num_heads, self.head_dim)
        K = K.reshape(batch, seq, self.num_heads, self.head_dim)
        V = V.reshape(batch, seq, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if self.use_ROPE:
            if token_positions == None:
                token_positions = torch.arange(seq, device = x.device).unsqueeze(0)
            
            token_positions = token_positions.unsqueeze(1)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(seq, seq, device = x.device, dtype = torch.bool))

        score =  scaled_dot_product_attention(Q, K, V, mask)
        score = score.transpose(1, 2)
        score = score.contiguous().reshape(batch, seq, d_model)

        out = self.output_proj(score)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            d_ff,
            max_seq_len,
            theta,
            device = None,
            dtype = None
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device = device, dtype = dtype)

        self.attn = MultiHeadAttention(
        d_model,
        num_heads,
        device=device,
        dtype=dtype,
        use_ROPE=True,
        theta=theta,
        max_seq_len=max_seq_len,
        )
        
        self.ln2 = RMSNorm(d_model, device = device, dtype = dtype)
        self.ffn = SwiGLU(d_model, d_ff, device = device, dtype = dtype)

    def forward(self, x, token_positions = None):

        normed = self.ln1(x)
        attn_out = self.attn(normed, token_positions = token_positions)
        x = x + attn_out

        #FFN
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
            d_ff,
            rope_theta,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device = device, dtype = dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                device = device,
                dtype = dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device = device, dtype = dtype)
        self.lm_head = Linear(d_model, vocab_size, device = device, dtype = dtype)
    
    def forward(self, in_indices):
        batch, seq = in_indices.shape
        
        token_positions = torch.arange(seq, device = in_indices.device).unsqueeze(0)

        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions = token_positions)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits

