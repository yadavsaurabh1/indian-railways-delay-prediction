"""Route transformer with cross-attention to the surrounding network.

  global token  (train, type, weekday, calendar, holidays, current delay, progress)
  + stop tokens (one per stop on the route: station, schedule relative to now,
                 observed delays so far, this train's history at that stop,
                 station congestion, weather)
      -> L x [self-attention over the route  ->  cross-attention to context  ->  FFN]
  context tokens (other trains near the next 6 halts + the rake-link pair),
      encoded by their own small transformer first.

Output per stop: 10th / 50th / 90th percentile of the delay (minutes), predicted
as an offset from the current known delay, with quantiles kept ordered.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from build_segments import K_SEG
from features import CTX_NUM, GLOB_NUM, STOP_NUM, TOK_NUM

QUANTILES = (0.1, 0.5, 0.9)


class MLPIn(nn.Module):
    def __init__(self, n_in: int, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention on F.scaled_dot_product_attention (fused kernel on MPS/CUDA).
    Parameter names match nn.MultiheadAttention, so older checkpoints load unchanged."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.d, self.h, self.p = d, heads, dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
        self.out_proj = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, q, kv, key_pad):
        B, Lq, d = q.shape
        S, h = kv.shape[1], self.h
        if q is kv:
            qq, k, v = F.linear(q, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        else:
            qq = F.linear(q, self.in_proj_weight[:d], self.in_proj_bias[:d])
            k, v = F.linear(kv, self.in_proj_weight[d:], self.in_proj_bias[d:]).chunk(2, dim=-1)
        qq = qq.view(B, Lq, h, d // h).transpose(1, 2)
        k = k.view(B, S, h, d // h).transpose(1, 2)
        v = v.view(B, S, h, d // h).transpose(1, 2)
        mask = (~key_pad)[:, None, None, :]  # True = attend
        o = F.scaled_dot_product_attention(qq, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        return self.out_proj(o.transpose(1, 2).reshape(B, Lq, d))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float, cross: bool, cross2: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = Attention(d, heads, dropout)
        self.cross = cross
        if cross:
            self.ln2 = nn.LayerNorm(d)
            self.ca = Attention(d, heads, dropout)
        self.cross2 = cross2
        if cross2:  # second memory: the other trains on our next stretches
            self.ln4 = nn.LayerNorm(d)
            self.ca2 = Attention(d, heads, dropout)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad, ctx=None, ctx_pad=None, tok=None, tok_pad=None):
        h = self.ln1(x)
        x = x + self.drop(self.sa(h, h, pad))
        if self.cross:
            h = self.ln2(x)
            x = x + self.drop(self.ca(h, ctx, ctx_pad))
        if self.cross2:
            h = self.ln4(x)
            x = x + self.drop(self.ca2(h, tok, tok_pad))
        return x + self.drop(self.ff(self.ln3(x)))


class RouteTransformer(nn.Module):
    def __init__(self, n_station: int, n_train: int, n_type: int, d: int = 192, heads: int = 6,
                 layers: int = 4, ctx_layers: int = 2, ff: int = 4, dropout: float = 0.1,
                 cumulative: bool = False, n_stop_num: int | None = None, tokens: bool = False):
        super().__init__()
        # checkpoints from before the stop-to-stop features have no n_stop_num: they used the first 43
        n_stop_num = n_stop_num or (len(STOP_NUM) if tokens else 43)
        self.config = dict(n_station=n_station, n_train=n_train, n_type=n_type, d=d, heads=heads,
                           layers=layers, ctx_layers=ctx_layers, ff=ff, dropout=dropout,
                           cumulative=cumulative, n_stop_num=n_stop_num, tokens=tokens)
        self.cumulative, self.tokens, self.n_stop_num = cumulative, tokens, n_stop_num
        self.station = nn.Embedding(n_station, d)
        self.train_emb = nn.Embedding(n_train, d)
        self.type_emb = nn.Embedding(n_type, d)
        self.dow = nn.Embedding(7, d)
        self.pos = nn.Embedding(256, d)
        self.slot = nn.Embedding(8, d)
        self.stop_in = MLPIn(n_stop_num, d)
        self.ctx_in = MLPIn(len(CTX_NUM), d)
        self.glob_in = MLPIn(len(GLOB_NUM), d)
        self.null_ctx = nn.Parameter(torch.zeros(1, 1, d))  # always-present token: "no other trains known"
        self.ctx_blocks = nn.ModuleList([Block(d, heads, ff, dropout, cross=False) for _ in range(ctx_layers)])
        if tokens:
            # other trains on our next stretches: numbers + kind (done / on it / opposite) + which stretch
            # + the stop it ends at + which train + its class
            self.tok_in = MLPIn(len(TOK_NUM), d)
            self.tok_kind = nn.Embedding(4, d)
            self.tok_k = nn.Embedding(K_SEG + 1, d)
            self.null_tok = nn.Parameter(torch.zeros(1, 1, d))
            self.tok_blocks = nn.ModuleList([Block(d, heads, ff, dropout, cross=False) for _ in range(ctx_layers)])
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout, cross=True, cross2=tokens) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # depth-scaled start (GPT-2 style): each block's writes into the residual stream start at
        # 1/sqrt(2 * depth) strength, so deep stacks begin near identity and train stably
        stacks = [self.blocks, self.ctx_blocks] + ([self.tok_blocks] if tokens else [])
        with torch.no_grad():
            for stack in stacks:
                s = (2 * len(stack)) ** -0.5
                for blk in stack:
                    outs = [blk.sa.out_proj, blk.ff[3]]
                    outs += [blk.ca.out_proj] if blk.cross else []
                    outs += [blk.ca2.out_proj] if blk.cross2 else []
                    for lin in outs:
                        lin.weight.mul_(s)

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        # context
        cc = b["ctx_cat"]
        ctx = (self.ctx_in(b["ctx_num"]) + self.slot(cc[..., 0]) + self.station(cc[..., 1])
               + self.train_emb(cc[..., 2]) + self.type_emb(cc[..., 3]))
        B = ctx.shape[0]
        ctx = torch.cat([self.null_ctx.expand(B, 1, -1), ctx], dim=1)
        ctx_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=ctx.device), b["ctx_pad"]], dim=1)
        for blk in self.ctx_blocks:
            ctx = blk(ctx, ctx_pad)

        tok = tok_pad = None
        if self.tokens:
            tc = b["tok_cat"]
            tok = (self.tok_in(b["tok_num"]) + self.tok_kind(tc[..., 0]) + self.tok_k(tc[..., 1])
                   + self.pos(tc[..., 2]) + self.train_emb(tc[..., 3]) + self.type_emb(tc[..., 4]))
            tok = torch.cat([self.null_tok.expand(B, 1, -1), tok], dim=1)
            tok_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tok.device), b["tok_pad"]], dim=1)
            for blk in self.tok_blocks:
                tok = blk(tok, tok_pad)

        # route
        gc = b["glob_cat"]
        glob = (self.glob_in(b["glob_num"]) + self.train_emb(gc[:, 0]) + self.type_emb(gc[:, 1])
                + self.dow(gc[:, 2])).unsqueeze(1)
        stops = self.stop_in(b["stop_num"][..., :self.n_stop_num]) + self.station(b["station"]) + self.pos(b["pos"])
        x = torch.cat([glob, stops], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), b["stop_pad"]], dim=1)
        for blk in self.blocks:
            x = blk(x, pad, ctx, ctx_pad, tok, tok_pad)
        out = self.head(self.ln_out(x[:, 1:])).float()  # [B, L, 3], in hours

        if self.cumulative:
            # stop-to-stop: out[..., 0] is the delay change on the stretch ending at this stop; the delay
            # at a stop is the current delay plus the changes on every stretch up to it. Passed stops and
            # padding contribute nothing.
            upcoming = ~b["stop_pad"] & (b["stop_num"][..., STOP_NUM.index("passed")] == 0)
            inc = torch.where(upcoming, out[..., 0], torch.zeros_like(out[..., 0]))
            mid = b["base"].unsqueeze(1) + 60.0 * torch.cumsum(inc, dim=1)
        else:
            mid = b["base"].unsqueeze(1) + 60.0 * out[..., 0]
        lo = mid - 60.0 * F.softplus(out[..., 1])
        hi = mid + 60.0 * F.softplus(out[..., 2])
        return torch.stack([lo, mid, hi], dim=-1)  # minutes


def pinball(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Quantile loss in hours, averaged over predicted stops."""
    q = torch.tensor(QUANTILES, device=pred.device).view(1, 1, 3)
    err = (target.unsqueeze(-1) - pred) / 60.0
    loss = torch.maximum(q * err, (q - 1) * err).sum(-1)
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp_min(1.0)
