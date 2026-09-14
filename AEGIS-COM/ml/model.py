"""Tiny causal speech-isolation net for Raspberry Pi-class CPUs.

Two heads, one shared GRU:
  gain  32 band gains  — keep speech, drop everything else
  vad   1 probability  — is a human talking in this 32 ms frame

The VAD head is what lets the pipeline output *only* speech: the mask cleans
the spectrum, the gate closes when nobody is talking. ~13k parameters.
Not a DeepFilterNet-scale model. Not analog ANC.
"""

from __future__ import annotations

import torch
from torch import nn

N_BANDS = 32
HIDDEN = 48


class TinyMaskNet(nn.Module):
    """Streaming GRU: one hop in → band gains + speech probability + hidden out."""

    def __init__(self, n_bands: int = N_BANDS, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.hidden = hidden
        self.gru = nn.GRU(n_bands, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, n_bands)
        self.fc_vad = nn.Linear(hidden, 1)

    def forward(self, bands: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, h_out = self.gru(bands, h)
        gain = torch.sigmoid(self.fc(y))
        vad = torch.sigmoid(self.fc_vad(y))
        return gain, vad, h_out

    def forward_seq(self, bands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, _ = self.gru(bands)
        return torch.sigmoid(self.fc(y)), torch.sigmoid(self.fc_vad(y))

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
