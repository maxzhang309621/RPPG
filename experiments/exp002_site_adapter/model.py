"""Shared PhysNet stem with a separate waveform head for each site."""
import torch
import torch.nn as nn

from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX


class SiteAdapter(nn.Module):
    def __init__(self, frames):
        super().__init__()
        source = PhysNet_padding_Encoder_Decoder_MAX(frames=frames)
        self.block8 = source.ConvBlock8
        self.block9 = source.ConvBlock9
        self.up1 = source.upsample
        self.up2 = source.upsample2
        self.pool = source.poolspa
        self.head = source.ConvBlock10

    def forward(self, features, length):
        x = self.block8(features)
        x = self.block9(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.pool(x)
        return self.head(x).view(-1, length)


class SiteObservers(nn.Module):
    """Early layers are shared. Face and palm never share the waveform head."""

    def __init__(self, frames):
        super().__init__()
        stem = PhysNet_padding_Encoder_Decoder_MAX(frames=frames)
        self.block1 = stem.ConvBlock1
        self.block2 = stem.ConvBlock2
        self.block3 = stem.ConvBlock3
        self.block4 = stem.ConvBlock4
        self.block5 = stem.ConvBlock5
        self.block6 = stem.ConvBlock6
        self.block7 = stem.ConvBlock7
        self.pool_spatial = stem.MaxpoolSpa
        self.pool_joint = stem.MaxpoolSpaTem
        self.face = SiteAdapter(frames)
        self.palm = SiteAdapter(frames)

    def encode(self, video):
        x = self.pool_spatial(self.block1(video))
        x = self.pool_joint(self.block3(self.block2(x)))
        x = self.pool_joint(self.block5(self.block4(x)))
        x = self.pool_spatial(self.block7(self.block6(x)))
        return x

    def forward(self, video, site):
        features = self.encode(video)
        adapter = self.face if site == "face" else self.palm
        return adapter(features, video.shape[2])
