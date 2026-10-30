"""SAGE-CT model: slice-aware Perceiver resampler, scan-report alignment heads, dual-context decoder with a
per-token knowledge gate."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NS = 24  # slices per study


def rope_angles(dim, grid, base=100.0):
    """3D rotary angles for (slice, row, col) on a 24 x grid x grid token lattice, head dim split into three parts."""
    parts = [dim // 3 // 2 * 2, dim // 3 // 2 * 2]
    parts.append(dim - sum(parts))
    pos = torch.stack(torch.meshgrid(torch.arange(NS), torch.arange(grid), torch.arange(grid), indexing="ij"), -1)
    pos = pos.reshape(-1, 3).float()
    return torch.cat([pos[:, a:a + 1] * (1.0 / (base ** (torch.arange(0, p, 2).float() / p)))[None]
                      for a, p in enumerate(parts)], -1)


def rotate(x, ang):
    """x (B, H, N, D), ang (N, D/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = ang.cos()[None, None], ang.sin()[None, None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


def ffn(d, drop):
    return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))


# ================================================================ encoder
class ResAttn(nn.Module):
    """attention of the resampler; optional rotary code on the keys."""

    def __init__(self, d, h, drop):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.drop = drop

    def forward(self, x, mem, k_ang=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(mem).view(B, mem.size(1), self.h, self.dh).transpose(1, 2)
        v = self.v(mem).view(B, mem.size(1), self.h, self.dh).transpose(1, 2)
        if k_ang is not None:
            k = rotate(k, k_ang)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class Resampler(nn.Module):
    """slice-cell tokens (B, 24 * grid * grid, in_dim) -> latents z (B, n_lat, d)."""

    def __init__(self, d=512, h=8, n_lat=64, blocks=2, drop=0.1, in_dim=3072, grid=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d))
        self.lat = nn.Parameter(torch.randn(n_lat, d) * 0.02)
        self.register_buffer("ang", rope_angles(d // h, grid), persistent=False)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.ModuleDict({"n1": nn.LayerNorm(d), "nm": nn.LayerNorm(d), "x": ResAttn(d, h, drop),
                                              "n2": nn.LayerNorm(d), "s": ResAttn(d, h, drop),
                                              "n3": nn.LayerNorm(d), "f": ffn(d, drop)}))
        self.out = nn.LayerNorm(d)

    def forward(self, f):
        mem = self.inp(f.float())
        z = self.lat[None].expand(f.size(0), -1, -1)
        for b in self.blocks:
            z = z + b["x"](b["n1"](z), b["nm"](mem), k_ang=self.ang)
            y = b["n2"](z)
            z = z + b["s"](y, y)
            z = z + b["f"](b["n3"](z))
        return self.out(z)


# ================================================================ decoder
class Attn(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h, self.dh, self.drop = h, d // h, drop
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x, mem, key_mask=None, causal=False):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(mem).view(B, mem.size(1), 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        m = None if key_mask is None else key_mask[:, None, None, :]  # True = attend
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=causal and m is None,
                                           dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class Layer(nn.Module):
    """masked self-attention, image and knowledge cross-attention, knowledge gate, FFN:
    h <- h + a_img + g * a_knw,  g = sigmoid(MLP([h; a_img; a_knw; cos(a_img, a_knw)]))."""

    def __init__(self, d, h, drop, gate="evidence"):
        super().__init__()
        self.n1, self.sa = nn.LayerNorm(d), Attn(d, h, drop)
        self.n2, self.xa, self.ka = nn.LayerNorm(d), Attn(d, h, drop), Attn(d, h, drop)
        self.gate_kind = gate
        if gate == "evidence":
            self.gate = nn.Sequential(nn.Linear(3 * d + 1, 128), nn.GELU(), nn.Linear(128, 1))
        elif gate == "state":  # ablation: gate from the decoder state only
            self.gate = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
        elif gate == "nocos":  # ablation: without the agreement term
            self.gate = nn.Sequential(nn.Linear(3 * d, 128), nn.GELU(), nn.Linear(128, 1))
        else:  # ablation "scalar": one learned scalar per layer (tanh, Flamingo-style)
            self.alpha = nn.Parameter(torch.zeros(1))
        self.n3, self.f = nn.LayerNorm(d), ffn(d, drop)
        self.dp = nn.Dropout(drop)

    def forward(self, x, z, r, r_mask):
        y = self.n1(x)
        x = x + self.dp(self.sa(y, y, causal=True))
        y = self.n2(x)
        a_img, a_knw = self.xa(y, z), self.ka(y, r, key_mask=r_mask)
        if self.gate_kind == "evidence":
            cos = F.cosine_similarity(a_img, a_knw, dim=-1).unsqueeze(-1)
            g = torch.sigmoid(self.gate(torch.cat([y, a_img, a_knw, cos], -1)))
        elif self.gate_kind == "state":
            g = torch.sigmoid(self.gate(y))
        elif self.gate_kind == "nocos":
            g = torch.sigmoid(self.gate(torch.cat([y, a_img, a_knw], -1)))
        else:
            g = torch.tanh(self.alpha).expand(*a_knw.shape[:-1], 1)
        x = x + self.dp(a_img) + self.dp(g * a_knw)
        return x + self.dp(self.f(self.n3(x)))


class SAGECT(nn.Module):
    def __init__(self, vocab, max_len, d=512, h=8, layers=3, drop=0.1, gate="evidence", in_dim=3072, grid=4,
                 n_rank=6, n_lat=64):
        super().__init__()
        self.res = Resampler(d, h, n_lat, 2, drop, in_dim, grid)
        # Stage 1: sigmoid contrastive scan-report alignment
        self.scan_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256))
        self.text_head = nn.Linear(768, 256)
        self.t_logit = nn.Parameter(torch.tensor(math.log(10.0)))
        self.t_bias = nn.Parameter(torch.tensor(-10.0))
        # knowledge tokens: clause embedding + rank of the retrieved case
        self.k_proj = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d))
        self.k_type = nn.Embedding(n_rank, d)
        # decoder
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len + 1, d)
        self.layers = nn.ModuleList([Layer(d, h, drop, gate) for _ in range(layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)
        self.null_z = nn.Parameter(torch.randn(1, 1, d) * 0.02)  # image dropout token
        self.no_rag = False  # ablation: no retrieved knowledge
        # auxiliary clause-latent alignment head, not used by the paper's configuration (no loss, no gradient);
        # kept so that the parameter count (Table IV) and the released checkpoints match
        self.tok_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256))
        self.fa_logit = nn.Parameter(torch.tensor(math.log(10.0)))
        self.fa_bias = nn.Parameter(torch.tensor(-10.0))

    # ---- Stage 1
    def scan_emb(self, z):
        return F.normalize(self.scan_head(z.mean(1)), dim=-1)

    def text_emb(self, e768):
        return F.normalize(self.text_head(e768), dim=-1)

    def siglip(self, z, rep):
        s, t = self.scan_emb(z), self.text_emb(rep)
        logits = s @ t.T * self.t_logit.exp() + self.t_bias
        y = 2 * torch.eye(len(s), device=s.device) - 1
        return -F.logsigmoid(y * logits).sum() / len(s)

    # ---- Stage 2
    def knowledge(self, E, T, M):
        """E (B, K, 768) clause embeddings, T (B, K) rank of the retrieved case, M (B, K) valid."""
        return self.k_proj(E) + self.k_type(T), M

    def forward(self, y, z, r, r_mask, drop_z=None):
        B = y.size(0)
        if drop_z is not None and drop_z.any():
            z = torch.where(drop_z[:, None, None], self.null_z.expand(B, z.size(1), -1), z)
        if self.no_rag:
            r_mask = torch.zeros_like(r_mask)
            r_mask[:, 0] = True
            r = torch.zeros_like(r)
        x = self.emb(y) + self.pos(torch.arange(y.size(1), device=y.device))[None]
        for lyr in self.layers:
            x = lyr(x, z, r, r_mask)
        return self.head(self.norm(x))
