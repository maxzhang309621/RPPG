"""FactorizePhys: Matrix Factorization for Multidimensional Attention
in Remote Physiological Sensing.

Reference:
    Joshi, J., Agaian, S. S., & Cho, Y. (2024). FactorizePhys: Matrix
    Factorization for Multidimensional Attention in Remote Physiological
    Sensing. NeurIPS 2024.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


# Default channel configuration
nf = [8, 12, 16]


# --------------------------------------------------------------------------- #
#  Matrix Decomposition Modules (NMF / VQ)
# --------------------------------------------------------------------------- #

class _MatrixDecompositionBase(nn.Module):
    def __init__(self, device, md_config, dim="3D"):
        super().__init__()
        self.dim = dim
        self.md_type = md_config["MD_TYPE"]
        if dim == "3D":
            self.transform = md_config["MD_TRANSFORM"]
        self.S = md_config["MD_S"]
        self.R = md_config["MD_R"]
        self.train_steps = md_config["MD_STEPS"]
        self.eval_steps = md_config["MD_STEPS"]
        self.inv_t = md_config["INV_T"]
        self.eta = md_config["ETA"]
        self.rand_init = md_config["RAND_INIT"]
        self.device = device

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        if self.dim == "3D":
            B, C, T, H, W = x.shape
            if self.transform.lower() == "t_kab":
                D = T // self.S
                N = C * H * W
            elif self.transform.lower() == "tk_ab":
                D = T * C // self.S
                N = H * W
            elif self.transform.lower() == "k_tab":
                D = C // self.S
                N = T * H * W
            else:
                raise ValueError(f"Invalid MD_TRANSFORM: {self.transform}")
            x = x.view(B * self.S, D, N)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)

        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1).to(self.device)

        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)

        # Reconstruct: (B*S, D, R) @ (B*S, N, R)^T -> (B*S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        if self.dim == "3D":
            x = x.view(B, C, T, H, W)

        bases = bases.view(B, self.S, D, self.R)
        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        return x

    @torch.no_grad()
    def online_update(self, bases):
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class NMF(_MatrixDecompositionBase):
    def __init__(self, device, md_config, dim="3D"):
        super().__init__(device, md_config, dim=dim)
        self.device = device
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.ones((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)
        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        # Update coefficients
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        # Update bases
        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef


class VQ(_MatrixDecompositionBase):
    def __init__(self, device, md_config, dim="3D"):
        super().__init__(device, md_config, dim=dim)
        self.device = device

    def _build_bases(self, B, S, D, R):
        bases = torch.ones((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)
        return bases

    @torch.no_grad()
    def local_step(self, x, bases, _):
        std_x = F.normalize(x, dim=1)
        std_bases = F.normalize(bases, dim=1, eps=1e-6)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        coef = coef / (1e-6 + coef.sum(dim=1, keepdim=True))
        bases = torch.bmm(x, coef)
        return bases, coef

    def compute_coef(self, x, bases, _):
        with torch.no_grad():
            x_norm = x.norm(dim=1, keepdim=True)
        std_x = x / (1e-6 + x_norm)
        std_bases = F.normalize(bases, dim=1, eps=1e-6)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        return coef


# --------------------------------------------------------------------------- #
#  Convolution Helpers
# --------------------------------------------------------------------------- #

class ConvBNReLU(nn.Module):
    @classmethod
    def _same_paddings(cls, kernel_size):
        if kernel_size == (1, 1, 1):
            return (0, 0, 0)
        elif kernel_size == (3, 3, 3):
            return (1, 1, 1)

    def __init__(self, in_c, out_c, kernel_size=1, stride=1, padding='same',
                 dilation=1, groups=1, act='relu', apply_bn=False, apply_act=True):
        super().__init__()
        self.apply_bn = apply_bn
        self.apply_act = apply_act
        dilation = (1, 1, 1) if dilation == 1 else dilation
        if kernel_size == 1:
            kernel_size = (1, 1, 1)
        if stride == 1:
            stride = (1, 1, 1)
        if padding == 'same':
            padding = self._same_paddings(kernel_size)

        self.conv = nn.Conv3d(in_c, out_c, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.ReLU(inplace=True)
        if self.apply_bn:
            self.bn = nn.InstanceNorm3d(out_c)

    def forward(self, x):
        x = self.conv(x)
        if self.apply_act:
            x = self.act(x)
        if self.apply_bn:
            x = self.bn(x)
        return x


# --------------------------------------------------------------------------- #
#  Features Factorization and Attention Module (FSAM)
# --------------------------------------------------------------------------- #

class FeaturesFactorizationModule(nn.Module):
    def __init__(self, inC, device, md_config):
        super().__init__()
        self.device = device
        md_type = md_config["MD_TYPE"]
        align_C = md_config["align_channels"]

        if "nmf" in md_type.lower():
            self.pre_conv_block = nn.Sequential(
                nn.Conv3d(inC, align_C, (1, 1, 1)),
                nn.ReLU(inplace=True))
        else:
            self.pre_conv_block = nn.Conv3d(inC, align_C, (1, 1, 1))

        if "nmf" in md_type.lower():
            self.md_block = NMF(self.device, md_config, dim="3D")
        elif "vq" in md_type.lower():
            self.md_block = VQ(self.device, md_config, dim="3D")
        else:
            raise ValueError(f"Unknown MD_TYPE: {md_type}")

        if "nmf" in md_type.lower():
            self.post_conv_block = nn.Sequential(
                ConvBNReLU(align_C, align_C, kernel_size=1),
                nn.Conv3d(align_C, inC, 1, bias=False))
        else:
            self.post_conv_block = nn.Sequential(
                ConvBNReLU(align_C, align_C, kernel_size=1, apply_act=False),
                nn.Conv3d(align_C, inC, 1, bias=False))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.pre_conv_block(x)
        att = self.md_block(x)
        dist = torch.dist(x, att)
        att = self.post_conv_block(att)
        return att, dist


# --------------------------------------------------------------------------- #
#  FactorizePhys Model Building Blocks
# --------------------------------------------------------------------------- #

class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride,
                      padding=padding, bias=False),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class rPPG_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1):
        super(rPPG_FeatureExtractor, self).__init__()
        #                                                       Input: B, inCh, T, 72, 72
        self.FeatureExtractor = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3,3,3], [1,1,1], [1,1,1]),  # B, 8,  T, 72, 72
            ConvBlock3D(nf[0], nf[1], [3,3,3], [1,2,2], [1,0,0]), # B, 12, T, 35, 35
            ConvBlock3D(nf[1], nf[1], [3,3,3], [1,1,1], [1,0,0]), # B, 12, T, 33, 33
            nn.Dropout3d(p=dropout_rate),
            ConvBlock3D(nf[1], nf[1], [3,3,3], [1,1,1], [1,0,0]), # B, 12, T, 31, 31
            ConvBlock3D(nf[1], nf[2], [3,3,3], [1,2,2], [1,0,0]), # B, 16, T, 15, 15
            ConvBlock3D(nf[2], nf[2], [3,3,3], [1,1,1], [1,0,0]), # B, 16, T, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        return self.FeatureExtractor(x)


class BVP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1):
        super(BVP_Head, self).__init__()
        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3,3,3], [1,1,1], [1,0,0]), # B, 16, T, 11, 11
            ConvBlock3D(nf[2], nf[2], [3,3,3], [1,1,1], [1,0,0]), # B, 16, T, 9, 9
            ConvBlock3D(nf[2], nf[2], [3,3,3], [1,1,1], [1,0,0]), # B, 16, T, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            inC = nf[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.final_layer = nn.Sequential(
            ConvBlock3D(nf[2], nf[1], [3,3,3], [1,1,1], [1,0,0]), # B, 12, T, 5, 5
            ConvBlock3D(nf[1], nf[0], [3,3,3], [1,1,1], [1,0,0]), # B, 8,  T, 3, 3
            nn.Conv3d(nf[0], 1, (3,3,3), stride=(1,1,1),
                      padding=(1,0,0), bias=False),                # B, 1,  T, 1, 1
        )

    def forward(self, voxel_embeddings, batch, length):
        voxel_embeddings = self.conv_block(voxel_embeddings)

        if (self.md_infer or self.training) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(
                    voxel_embeddings - voxel_embeddings.min())
            else:
                att_mask, appx_error = self.fsam(voxel_embeddings)

            if self.md_res:
                x = torch.mul(
                    voxel_embeddings - voxel_embeddings.min() + self.bias1,
                    att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)
                factorized_embeddings = voxel_embeddings + factorized_embeddings
            else:
                x = torch.mul(
                    voxel_embeddings - voxel_embeddings.min() + self.bias1,
                    att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)

            x = self.final_layer(factorized_embeddings)
        else:
            x = self.final_layer(voxel_embeddings)

        rPPG = x.view(-1, length)

        if (self.md_infer or self.training) and self.use_fsam:
            return rPPG, factorized_embeddings, appx_error
        else:
            return rPPG


# --------------------------------------------------------------------------- #
#  Main FactorizePhys Model
# --------------------------------------------------------------------------- #

# Default config values (filled in if not provided by user config)
_default_md_config = {
    "MD_FSAM": True,
    "MD_TYPE": "NMF",
    "MD_TRANSFORM": "T_KAB",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 4,
    "MD_INFERENCE": False,
    "MD_RESIDUAL": False,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True,
    "in_channels": 3,
    "align_channels": nf[2] // 2,
}


class FactorizePhys(nn.Module):
    """FactorizePhys model (standard variant, 72x72 spatial input).

    Applies temporal differencing internally: input of T+1 frames produces
    T-frame output. The trainer is responsible for appending the extra frame.

    Args:
        frames: Number of output frames (T). Input should be T+1 frames.
        md_config: Dict of matrix decomposition / FSAM configuration.
        in_channels: Number of input channels (default 3 for RGB).
        dropout: Dropout rate for Dropout3d layers.
        device: torch.device for FSAM internal buffers.
    """

    def __init__(self, frames=128, md_config=None, in_channels=3,
                 dropout=0.1, device=torch.device("cpu")):
        super(FactorizePhys, self).__init__()

        if md_config is None:
            md_config = {}

        # Fill defaults for any missing config keys
        for key in _default_md_config:
            if key not in md_config:
                md_config[key] = _default_md_config[key]

        self.in_channels = in_channels
        self.use_fsam = md_config["MD_FSAM"]
        self.md_infer = md_config["MD_INFERENCE"]

        self.norm = nn.InstanceNorm3d(self.in_channels)

        self.rppg_feature_extractor = rPPG_FeatureExtractor(
            self.in_channels, dropout_rate=dropout)
        self.rppg_head = BVP_Head(
            md_config, device=device, dropout_rate=dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, T+1, H, W) input tensor.

        Returns:
            If FSAM active (training or md_infer):
                (rPPG, voxel_embeddings, factorized_embeddings, appx_error)
            Else:
                (rPPG, voxel_embeddings)
            where rPPG has shape (B, T).
        """
        [batch, channel, length, width, height] = x.shape

        # Temporal differencing: T+1 -> T
        x = torch.diff(x, dim=2)

        # Instance normalization
        x = self.norm(x[:, :self.in_channels, :, :, :])

        # Feature extraction
        voxel_embeddings = self.rppg_feature_extractor(x)

        # BVP prediction head
        if (self.md_infer or self.training) and self.use_fsam:
            rPPG, factorized_embeddings, appx_error = self.rppg_head(
                voxel_embeddings, batch, length - 1)
            return rPPG, voxel_embeddings, factorized_embeddings, appx_error
        else:
            rPPG = self.rppg_head(voxel_embeddings, batch, length - 1)
            return rPPG, voxel_embeddings
