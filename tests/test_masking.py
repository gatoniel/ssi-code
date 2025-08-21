from ssi_code.models.masking import Masking
from torch import nn
import torch


def test_masking_backwards():
    model = nn.Conv2d(2, 1, kernel_size=3, padding="same")
    masking = Masking(model)

    x = torch.randn(1, 2, 16, 16)

    masking.train()

    y = masking(x)

    loss = (y * masking.mask).sum()

    loss.backward()
