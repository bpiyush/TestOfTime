"""Video-Language Contrastive Losses"""

import numpy as np
import torch
from torch import nn

from external.fairseq.examples.MMPT.mmpt.losses.nce import Loss


# Loss composer: albegraic combination of losses
class LossAddition:
    def __init__(self, losses: list) -> None:
        # self.losses = [loss() for loss in losses]
        self.losses = losses
    def __call__(self, **kwds):
        loss = 0.
        for l in self.losses:
            loss += l(**kwds)
        return loss


class T2VContraLoss(Loss):
    """NCE for MM joint space, on softmax text2video matrix.
    """
    def __init__(self, weight=None, alpha=1.0):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.alpha = np.log(alpha + 1e-8)
        if isinstance(self.alpha, np.ndarray):
            self.alpha = torch.from_numpy(self.alpha)

    def __call__(self, pooled_video, pooled_text, **kargs):
        batch_size = pooled_video.size(0)
        # change device of the weight
        self.loss.weight = self.loss.weight.to(pooled_video.device)
        self.alpha = self.alpha.to(pooled_video.device)
        logits = torch.mm(pooled_text, pooled_video.transpose(1, 0))
        logits[batch_size // 2:, :batch_size // 2] += self.alpha
        logits[:batch_size // 2, batch_size // 2:] += self.alpha
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=pooled_video.device)
        return self.loss(logits, targets)


class V2TContraLoss(Loss):
    """NCE for MM joint space, with softmax on video2text matrix."""

    def __init__(self, weight=None, alpha=1.0):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.alpha = np.log(alpha + 1e-8)
        if isinstance(self.alpha, np.ndarray):
            self.alpha = torch.from_numpy(self.alpha)

    def __call__(self, pooled_video, pooled_text, **kargs):
        batch_size = pooled_video.size(0)
        # change device of the weight
        self.loss.weight = self.loss.weight.to(pooled_video.device)
        self.alpha = self.alpha.to(pooled_video.device)
        logits = torch.mm(pooled_video, pooled_text.transpose(1, 0))
        logits[batch_size // 2:, :batch_size // 2] += self.alpha
        logits[:batch_size // 2, batch_size // 2:] += self.alpha
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=pooled_video.device)
        return self.loss(logits, targets)
