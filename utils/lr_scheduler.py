import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineWithLinearWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(CosineWithLinearWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase: from min_lr to max_lr
            warmup_lr = [
                self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch / self.warmup_steps
                for _ in self.base_lrs
            ]
            return warmup_lr
        else:
            # Cosine decay phase: from max_lr back to min_lr
            decay_step = self.last_epoch - self.warmup_steps
            total_decay_steps = self.total_steps - self.warmup_steps
            cosine_decay_lr = [
                self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * decay_step / total_decay_steps)) / 2
                for _ in self.base_lrs
            ]
            return cosine_decay_lr