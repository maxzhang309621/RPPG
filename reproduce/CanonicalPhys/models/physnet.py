"""PhysNet: Remote Photoplethysmograph Signal Measurement from Facial Videos
Using Spatio-Temporal Networks.

Reference:
    Yu, Z., Li, X., & Zhao, G. (2019). Remote Photoplethysmograph Signal
    Measurement from Facial Videos Using Spatio-Temporal Networks.
    British Machine Vision Conference (BMVC).
"""

import torch
import torch.nn as nn


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128, drop_rate=0.0):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout3d(p=drop_rate)

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):  # (B, 3, T, H, W)
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)       # (B, 16, T, H, W)
        x = self.MaxpoolSpa(x)       # (B, 16, T, H/2, W/2)

        x = self.ConvBlock2(x)       # (B, 32, T, H/2, W/2)
        x = self.ConvBlock3(x)       # (B, 64, T, H/2, W/2)
        x = self.MaxpoolSpaTem(x)    # (B, 64, T/2, H/4, W/4)
        x = self.dropout(x)

        x = self.ConvBlock4(x)       # (B, 64, T/2, H/4, W/4)
        x = self.ConvBlock5(x)       # (B, 64, T/2, H/4, W/4)
        x = self.MaxpoolSpaTem(x)    # (B, 64, T/4, H/8, W/8)
        x = self.dropout(x)

        x = self.ConvBlock6(x)       # (B, 64, T/4, H/8, W/8)
        x = self.ConvBlock7(x)       # (B, 64, T/4, H/8, W/8)
        x = self.MaxpoolSpa(x)       # (B, 64, T/4, H/16, W/16)
        x = self.dropout(x)

        x = self.ConvBlock8(x)       # (B, 64, T/4, H/16, W/16)
        x = self.ConvBlock9(x)       # (B, 64, T/4, H/16, W/16)
        x = self.upsample(x)         # (B, 64, T/2, H/16, W/16)
        x = self.upsample2(x)        # (B, 64, T, H/16, W/16)

        x = self.poolspa(x)          # (B, 64, T, 1, 1)
        x = self.ConvBlock10(x)      # (B, 1, T, 1, 1)

        rPPG = x.view(-1, length)    # (B, T)

        return rPPG
