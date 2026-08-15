"""Restormer — clean-room reimplementation.

See ATTRIBUTION.md in this directory for why this is a reimplementation
rather than vendored official code, and for the citation. Paper: Zamir et
al., "Restormer: Efficient Transformer for High-Resolution Image
Restoration", CVPR 2022.

Key ideas (from the paper's published description):
    - MDTA: channel-wise (not spatial) self-attention -- linear cost in
      image resolution instead of quadratic, via depth-wise-conv-derived
      Q/K/V and an attention map over channels-per-head.
    - GDFN: GELU-gated feed-forward network with depth-wise convs for
      local context.
    - 4-level U-Net of Transformer blocks, PixelUnshuffle/PixelShuffle for
      down/upsampling, concat+1x1-reduce skip connections, refinement
      stage at full resolution, global residual connection.

Adaptation for this project (as with NAFNet): an SR head is appended for
scale_factor > 1, since the original architecture is same-resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.registry import MODEL_REGISTRY
from models.base import BaseRestorationModel


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors (same approach as
    models/nafnet/model.py's LayerNorm2d — this architectural piece is
    common to both papers' published designs, not copied between files)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention: channel-wise self-attention."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        # (B, C, H, W) -> (B, num_heads, head_dim, H*W)
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Attention over CHANNELS (head_dim x head_dim), not over spatial positions --
        # this is the linear-in-resolution trick the paper is built around.
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, num_heads, head_dim, H*W)

        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False), nn.PixelUnshuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False), nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _make_stage(dim: int, num_heads: int, num_blocks: int, ffn_expansion_factor: float) -> nn.Sequential:
    return nn.Sequential(*[TransformerBlock(dim, num_heads, ffn_expansion_factor) for _ in range(num_blocks)])


@MODEL_REGISTRY.register("restormer")
class Restormer(BaseRestorationModel):
    def __init__(
        self,
        scale_factor: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 24,
        num_blocks: list[int] | None = None,
        num_refinement_blocks: int = 1,
        heads: list[int] | None = None,
        ffn_expansion_factor: float = 2.66,
    ) -> None:
        super().__init__(scale_factor=scale_factor, in_channels=in_channels, out_channels=out_channels)
        num_blocks = num_blocks if num_blocks is not None else [1, 1, 1, 1]
        heads = heads if heads is not None else [1, 2, 4, 8]
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("num_blocks and heads must each have exactly 4 entries (one per U-Net level)")

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)

        self.encoder_level1 = _make_stage(dim, heads[0], num_blocks[0], ffn_expansion_factor)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = _make_stage(dim * 2, heads[1], num_blocks[1], ffn_expansion_factor)
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = _make_stage(dim * 4, heads[2], num_blocks[2], ffn_expansion_factor)
        self.down3_4 = Downsample(dim * 4)
        self.latent = _make_stage(dim * 8, heads[3], num_blocks[3], ffn_expansion_factor)  # bottleneck

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
        self.decoder_level3 = _make_stage(dim * 4, heads[2], num_blocks[2], ffn_expansion_factor)

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.decoder_level2 = _make_stage(dim * 2, heads[1], num_blocks[1], ffn_expansion_factor)

        self.up2_1 = Upsample(dim * 2)
        # Finest level keeps the concatenated dim*2 channels (no reduction), per the
        # paper's published design -- decoder_level1 and refinement both operate at dim*2.
        self.decoder_level1 = _make_stage(dim * 2, heads[0], num_blocks[0], ffn_expansion_factor)
        self.refinement = _make_stage(dim * 2, heads[0], num_refinement_blocks, ffn_expansion_factor)

        if scale_factor > 1:
            self.sr_head = nn.Sequential(
                nn.Conv2d(dim * 2, out_channels * scale_factor**2, kernel_size=3, padding=1),
                nn.PixelShuffle(scale_factor),
            )
            self.output = None
        else:
            self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1, bias=False)
            self.sr_head = None

        self.padder_size = 8  # 3 downsample stages -> spatial dims must be multiples of 2^3

    def _check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        inp = self._check_image_size(x)

        feat1 = self.patch_embed(inp)
        enc1 = self.encoder_level1(feat1)

        feat2 = self.down1_2(enc1)
        enc2 = self.encoder_level2(feat2)

        feat3 = self.down2_3(enc2)
        enc3 = self.encoder_level3(feat3)

        feat4 = self.down3_4(enc3)
        latent = self.latent(feat4)

        dec3 = self.up4_3(latent)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.reduce_chan_level3(dec3)
        dec3 = self.decoder_level3(dec3)

        dec2 = self.up3_2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.reduce_chan_level2(dec2)
        dec2 = self.decoder_level2(dec2)

        dec1 = self.up2_1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # no channel reduction at the finest level
        dec1 = self.decoder_level1(dec1)
        dec1 = self.refinement(dec1)

        if self.sr_head is not None:
            out = self.sr_head(dec1)
            upsampled_input = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            out = out[:, :, : h * self.scale_factor, : w * self.scale_factor] + upsampled_input
        else:
            out = self.output(dec1) + inp
            out = out[:, :, :h, :w]

        return out
