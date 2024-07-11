"""Misc. optimizer implementations."""
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR

# 调整训练过程中的学习率
# CosineAnnealingLR 是另一种调度器，它按照余弦函数的形式调整学习率。
# 而 One-Cycle 调度器是一种在训练过程中先增加后减少学习率的策略，通常用于加速训练并提高模型性能。

def get_schedule_fn(scheduler, num_training_steps):
    '''Returns a callable scheduler_fn(optimizer).
    Todo: Sanitize and unify these schedulers...

    Args:
        scheduler:scheduler 是一个字符串，指定了要使用的调度器类型
        num_training_steps:num_training_steps 是训练过程中的总步数

    Returns: 一个新的函数

    '''

    if scheduler == "cosine-decay":
        scheduler_fn = partial(
            torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=num_training_steps,
            eta_min=0.0,
        )
    elif scheduler == "one-cycle":  # this is a simplified one-cycle
        scheduler_fn = partial(
            get_one_cycle,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Invalid schedule {scheduler} given.")
    return scheduler_fn


def get_one_cycle(optimizer, num_training_steps):
    """Simple single-cycle scheduler. Not including paper/fastai three-phase things or asymmetry."""

    def lr_lambda(current_step):
        if current_step < num_training_steps / 2:
            return float(current_step / (num_training_steps / 2))
        else:
            return float(2 - current_step / (num_training_steps / 2))

    return LambdaLR(optimizer, lr_lambda, -1)
