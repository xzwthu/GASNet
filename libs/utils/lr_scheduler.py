#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   09 January 2019


from torch.optim.lr_scheduler import (
    _LRScheduler, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
)


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, step_size, iter_max, power=0.9, last_epoch=-1):
        self.step_size = step_size
        self.iter_max = iter_max
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(
        self, optimizer, scheduler, mode="linear", warmup_iters=100, gamma=0.2, last_epoch=-1
    ):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_iters:
            if self.mode == "linear":
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == "constant":
                factor = self.gamma
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs


key2scheduler = {
    "constant_lr": ConstantLR,
    "poly_lr": PolynomialLR,
    "step_lr": StepLR,
    "multi_step": MultiStepLR,
    "cosine_annealing": CosineAnnealingLR,
    "exp_lr": ExponentialLR,
}


def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        return ConstantLR(optimizer)
        
    s_type = scheduler_dict["name"]
    scheduler_dict.pop("name")

    warmup_dict = {}
    if "warmup_iters" in scheduler_dict:
        # This can be done in a more pythonic way...
        warmup_dict["warmup_iters"] = scheduler_dict.get("warmup_iters", 100)
        warmup_dict["mode"] = scheduler_dict.get("warmup_mode", "linear")
        warmup_dict["gamma"] = scheduler_dict.get("warmup_factor", 0.2)

        scheduler_dict.pop("warmup_iters", None)
        scheduler_dict.pop("warmup_mode", None)
        scheduler_dict.pop("warmup_factor", None)

        base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)

    return key2scheduler[s_type](optimizer, **scheduler_dict)
    

