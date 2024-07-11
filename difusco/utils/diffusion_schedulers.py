"""Schedulers for Denoising Diffusion Probabilistic Models"""

import math

import numpy as np
import torch
from ..models import NoiseSchedule  # 用作动态噪声计划


class GaussianDiffusion(object):
    """Gaussian Diffusion process with linear beta scheduling"""

    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T

        # Noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)  # 一维数组 长度为T
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
                0)  # Generate an extra alpha for bT
            # self.beta 数组将具有 T + 1 个元素，但经过 np.clip 后，实际上使用的是从 1 到 T 的 beta 值
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)  # 一维数组 长度为T
        elif schedule == 'nn':
            self.alphabar = self.__nn_noise(np.arange(0, T + 1, 1)) / self.__nn_noise(
                0)
            # 这里要对nn产生的噪声有一个范围 这里的范围有问题
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)# 生成的噪声变为对应的形状 在方法中产生的噪声本身就是一个数组形式

        # cumprod 计算出来的是每一步的噪声水平 用betabar代替
        self.betabar = np.cumprod(self.beta)
        # 将 beta 理解为每一步中添加的噪声的比例，那么 1 - beta 就是原始数据在噪声添加后保留的比例
        # alpha 值实际上是定义了每一步中数据的方差相对于前一步的减少量，或者说是数据的“方差减少因子，就可以通过这个数组 后续完成逆向去噪
        self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))  # 计算α
        self.alphabar = np.cumprod(self.alpha)  # 计算出来的是每一步的原始数据水平 用alphabar代替

    def __cos_noise(self, t):
        # 生成余弦调度的噪声水平。它使用了一个偏移量 offset 来调整余弦波形。
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def __nn_noise(self, t):
        # 由神经网络生成噪声
        nn_noise = NoiseSchedule.NoiseSchedule_NNet()
        _nn_noise = nn_noise.__call__(self.t)
        return _nn_noise

    def sample(self, x0, t):
        # Select noise scales
        # 这个方法接受原始数据 x0 和时间步 t，然后根据当前步的 alphabar 值和随机噪声 epsilon 生成扩散后的数据 xt
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
        atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

        # Sample noise and add to x0
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
        return xt, epsilon #扩散后的数据 xt 和生成的随机噪声 epsilon


class CategoricalDiffusion(object):
    """Gaussian Diffusion process with linear beta scheduling"""

    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T

        # Noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
                0)  # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))

        self.Qs = (1 - beta) * eye + (beta / 2) * ones

        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar, axis=0)

    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def sample(self, x0_onehot, t):
        # Select noise scales
        Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
        xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
        return torch.bernoulli(xt[..., 1].clamp(0, 1))


class InferenceSchedule(object):
    '''
    用来加速推断速度 就是加速去噪过程，直接的方式就是减少反向过程的步骤数
    '''

    def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T

    def __call__(self, i):
        assert 0 <= i < self.inference_T

        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "nn":  # 可学习网络的推理过程
            '''
            可改进 这里的t1和t2对应的应该分别是起始时间
            '''
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)  # 保证合适的范围

            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)  # 保证合适的范围
            return t1, t2
        else:
            raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))
