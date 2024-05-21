# -*- coding:utf-8 -
import torch
import torch.nn as nn
from modules.EfficientAT.models.dymn.model import get_model as get_dymn
from modules.EfficientAT.models.preprocess import AugmentMelSTFT
from einops.layers.torch import Rearrange


class AudioSleepNet(nn.Module):
    def __init__(self, seq_len=40, rnn_hidden=128, output_classes=None):
        super().__init__()
        self.dropout_rate = 0.5
        self.pretrained_name = 'dymn10_as'
        self.seq_len = seq_len
        self.backbone = self.get_backbone()
        self.backbone_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.backbone.lastconv_output_channels, self.backbone.last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(self.backbone.last_channel, self.backbone.last_channel // 4),
        )
        self.rnn = nn.LSTM(input_size=self.backbone.last_channel // 4,
                           hidden_size=rnn_hidden, bidirectional=True)
        self.classifiers = nn.ModuleDict({
            output_name: nn.Sequential(
                nn.Linear(rnn_hidden * 2, rnn_hidden),
                Rearrange('b t e -> b e t'),
                nn.BatchNorm1d(rnn_hidden),
                nn.ELU(),
                Rearrange('b e t -> b t e'),
                nn.Linear(rnn_hidden, num_class)
            )
            for output_name, num_class in output_classes.items()
        })

    def forward(self, x):
        outs = []
        for x_split in torch.split(x, dim=1, split_size_or_sections=1):
            o = self.backbone.feature_forward(x_split)
            o = self.backbone_projector(o)
            outs.append(o)
        backbone_out = torch.stack(outs, dim=1)
        rnn_out, _ = self.rnn(backbone_out)

        outs = {}
        for output_name, classifier in self.classifiers.items():
            out = classifier(rnn_out)
            outs[output_name] = out
        return outs

    def get_backbone(self):
        backbone = get_dymn(pretrained_name=self.pretrained_name)
        del backbone.classifier
        return backbone


if __name__ == '__main__':
    import os
    import torch
    import numpy as np

    mel = AugmentMelSTFT(n_mels=128, sr=48000, win_length=800, hopsize=320)
    temp = []
    for _ in range(40):
        signal = torch.randn(50, 48000)
        spec = mel(signal)
        temp.append(spec)
    temp = torch.stack(temp, dim=1)
    print(temp.shape)

    net = AudioSleepNet(seq_len=40, output_classes={
        'sleep_staging': 4,
        'apnea': 2,
        'hypopnea': 2,
        'arousal': 2,
        'snore': 2
    })
    outs_ = net(
        temp
    )

