"""EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based
Cardiac Measurement.

Reference:
    X. Liu, B. Hill, Z. Jiang, S. Patel, D. McDuff.
    "EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based
    Vitals Measurement." WACV 2023.

Architecture:
    2D CNN with Temporal Shift Module (TSM) for temporal modeling and
    spatial attention gating for appearance-guided motion estimation.
    The model operates on individual frames (N*T, C, H, W) and uses
    TSM to shift features across the temporal dimension.

    Applies torch.diff internally: T+1 input frames → T output values.
"""

import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    """Spatial attention normalization."""

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5


class TSM(nn.Module):
    """Temporal Shift Module.

    Shifts a fraction of channels forward/backward in time to capture
    temporal dependencies without additional parameters or computation.

    Args:
        n_segment: Number of frames per temporal group.
        fold_div: Denominator for channel fraction to shift.
    """

    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]            # shift left
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]        # no shift

        return out.view(nt, c, h, w)


class EfficientPhys(nn.Module):
    """EfficientPhys model with TSM and spatial attention gating.

    Args:
        in_channels: Number of input channels (default 3).
        nb_filters1: Channels in first conv block (default 32).
        nb_filters2: Channels in second conv block (default 64).
        kernel_size: Convolution kernel size (default 3).
        dropout_rate1: Spatial dropout rate (default 0.25).
        dropout_rate2: Dense dropout rate (default 0.5).
        pool_size: Spatial pooling kernel (default (2,2)).
        nb_dense: Dense layer width (default 128).
        frame_depth: TSM segment size (default 20).
        img_size: Spatial input size (36, 72, or 96).
    """

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64,
                 kernel_size=3, dropout_rate1=0.25, dropout_rate2=0.5,
                 pool_size=(2, 2), nb_dense=128, frame_depth=20,
                 img_size=72):
        super(EfficientPhys, self).__init__()

        self.in_channels = in_channels
        self.frame_depth = frame_depth

        # Batch normalization for frame differences
        self.batch_norm = nn.BatchNorm2d(in_channels)

        # TSM modules
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)

        # Motion branch 1 (conv1 with padding, conv2 without)
        self.motion_conv1 = nn.Conv2d(
            in_channels, nb_filters1, kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(
            nb_filters1, nb_filters1, kernel_size, bias=True)  # no padding

        # Motion branch 2 (conv3 with padding, conv4 without)
        self.motion_conv3 = nn.Conv2d(
            nb_filters1, nb_filters2, kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(
            nb_filters2, nb_filters2, kernel_size, bias=True)  # no padding

        # Appearance attention (1x1 conv → sigmoid → normalize)
        # att_conv1: nb_filters1 → 1 channel
        self.apperance_att_conv1 = nn.Conv2d(
            nb_filters1, 1, kernel_size=1, padding=0, bias=True)
        self.attn_mask_1 = Attention_mask()
        # att_conv2: nb_filters2 → 1 channel
        self.apperance_att_conv2 = nn.Conv2d(
            nb_filters2, 1, kernel_size=1, padding=0, bias=True)
        self.attn_mask_2 = Attention_mask()

        # Pooling (3 defined, only avg_pooling_1 and avg_pooling_3 used)
        self.avg_pooling_1 = nn.AvgPool2d(pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(pool_size)

        # Dropout (4 defined, dropout_1/3/4 used in forward)
        self.dropout_1 = nn.Dropout(dropout_rate1)
        self.dropout_2 = nn.Dropout(dropout_rate1)
        self.dropout_3 = nn.Dropout(dropout_rate1)
        self.dropout_4 = nn.Dropout(dropout_rate2)

        # Dense layers
        # Spatial dimensions after forward: img → conv1(pad1) → conv2(nopad) → pool → conv3(pad1) → conv4(nopad) → pool
        # For img_size=72: 72 → 72 → 70 → 35 → 35 → 33 → 16 → flat: 64*16*16=16384
        if img_size == 36:
            dense_input = 3136
        elif img_size == 72:
            dense_input = 16384
        elif img_size == 96:
            dense_input = 30976
        else:
            raise ValueError(f"Unsupported img_size: {img_size}")

        self.final_dense_1 = nn.Linear(dense_input, nb_dense, bias=True)
        self.final_dense_2 = nn.Linear(nb_dense, 1, bias=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (N*T_in, C, H, W) — flattened batch of frames.
               T_in includes 1 extra frame for temporal differencing.

        Returns:
            out: (N*T_out, 1) per-frame BVP predictions, where T_out = T_in - 1.
        """
        # Temporal differencing: N*T_in → N*(T_in-1)
        x = torch.diff(x, dim=0)
        x = self.batch_norm(x)

        # --- Block 1: TSM → Conv → Attention gating ---
        d1 = self.TSM_1(x)
        d1 = torch.tanh(self.motion_conv1(d1))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        # Spatial attention on motion features
        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        # --- Block 2: TSM → Conv → Attention gating ---
        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)

        # --- Dense head ---
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out
