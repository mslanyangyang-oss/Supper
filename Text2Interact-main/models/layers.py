
import sys
sys.path.append(sys.path[0] + r"/../")
from .utils import zero_module
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class AdaLN(nn.Module):

    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale[:, None]) + shift[:, None]
        return h


class VanillaSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout, batch_first=True,
                                               add_zero_attn=True)

    def forward(self, x, emb, key_padding_mask=None):
        """
        x: B, T, D
        """
        x_norm = self.norm(x, emb)
        y = self.attention(x_norm, x_norm, x_norm,
                           attn_mask=None,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return y


class VanillaCrossAttention(nn.Module):

    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(latent_dim, num_head, kdim=xf_latent_dim, vdim=xf_latent_dim,
                                               dropout=dropout, batch_first=True, add_zero_attn=True)

    def forward(self, x, xf, emb, key_padding_mask=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        x_norm = self.norm(x, emb)
        xf_norm = self.xf_norm(xf, emb)
        y = self.attention(x_norm, xf_norm, xf_norm,
                           attn_mask=None,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return y


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, embed_dim=None):
        super().__init__()
        self.norm = AdaLN(latent_dim, embed_dim)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim, bias=True))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        if emb is not None:
            x_norm = self.norm(x, emb)
        else:
            x_norm = x
        y = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        return y


class FinalLayer(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.linear = zero_module(nn.Linear(latent_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.linear(x)
        return x

class SelfAttention(nn.Module):

    def __init__(
        self,
        latent_dim,
        text_latent_dim,
        num_heads: int = 8,
        dropout: float = 0.0,
        log_attn=False,
        embed_dim=None,
        index=None
    ):
        super().__init__()
        self.num_head = num_heads
        self.norm = AdaLN(latent_dim, embed_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=True)

        self.log_attn = log_attn
        self.index = index
        if index is None:
            raise ValueError("index must be provided for editable self attention")

    def forward(self, x, emb, key_padding_mask=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = x.shape[1]
        assert N == T
        H = self.num_head

        # B, T, 1, D
        query = self.query(self.norm(x, emb)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.norm(x, emb)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)

        # B, T, N, H
        attention = torch.einsum("bnhd,bmhd->bnmh", query, key) / math.sqrt(D // H)

        if key_padding_mask is not None:
            # Mask shape: (B, 1, N, 1) → broadcast to (B, T, N, H)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            attention = attention.masked_fill(mask, float('-inf'))

        weight = self.dropout(F.softmax(attention, dim=2))

        value = self.value(self.norm(x, emb)).view(B, N, H, -1)

        y = torch.einsum("bnmh,bmhd->bnhd", weight, value).reshape(B, T, D)
        y = self.out_proj(y)
        return y


class CrossAttention(nn.Module):

    def __init__(
        self,
        latent_dim,
        text_latent_dim,
        num_heads: int = 8,
        dropout: float = 0.0,
        log_attn=False,
        embed_dim=None,
        index=None
    ):
        super().__init__()
        self.num_head = num_heads
        self.norm = AdaLN(latent_dim, embed_dim)
        self.text_norm = AdaLN(text_latent_dim, embed_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=True)

        self.log_attn = log_attn


    def forward(self, x, xf, emb, key_padding_mask=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x, emb)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf, emb)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum("bnhd,bmhd->bnmh", query, key) / math.sqrt(D // H)

        # Apply key_padding_mask
        if key_padding_mask is not None:
            # Mask shape: (B, 1, N, 1) → broadcast to (B, T, N, H)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            attention = attention.masked_fill(mask, float('-inf'))

        weight = self.dropout(F.softmax(attention, dim=2))

        value = self.value(self.text_norm(xf, emb)).view(B, N, H, -1)
        y = torch.einsum("bnmh,bmhd->bnhd", weight, value).reshape(B, T, D)
        y = self.out_proj(y)
        return y





