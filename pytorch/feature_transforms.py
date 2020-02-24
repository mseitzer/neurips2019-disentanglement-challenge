import gin
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def l2_normalize(x):
    l2_norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
    # This is awkward, but ensures that this tensor lands on the correct
    # device during evaluation. The problem is that while tracing, a fixed
    # device is chosen when using `torch.ones_like(l2_norm)`, and this device
    # might be different from the evaluation device.
    ones = (l2_norm - l2_norm + 1.0)
    l2_norm = torch.where(l2_norm > 0.0, l2_norm, ones)
    return x / l2_norm


def to_bhwc(x):
    B, C, H, W = x.shape
    return x.view(B, C, -1).transpose(1, 2).view(B, H, W, C).contiguous()


class FeatureTransform(nn.Module):
    def __init__(self, l2_normalize, pca=None):
        super(FeatureTransform, self).__init__()
        self.l2_normalize = l2_normalize
        self.pca = pca

    def transform(self, x):
        """Transform 3D feature maps into 1D vectors"""
        raise NotImplementedError('Must be implemented in child classes')

    def forward(self, x):
        x = self.transform(x)

        if self.l2_normalize:
            x = l2_normalize(x)

        # Make sure we return something 2D
        return torch.reshape(x, (x.shape[0], -1))


@gin.configurable
class IdentityTransform(FeatureTransform):
    def __init__(self, l2_normalize=False):
        super(IdentityTransform, self).__init__(l2_normalize)

    def transform(self, x):
        return x


@gin.configurable
class AverageTransform(FeatureTransform):
    def __init__(self, l2_normalize=True):
        super(AverageTransform, self).__init__(l2_normalize)

    def transform(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))


@gin.configurable
class MaximumTransform(FeatureTransform):
    def __init__(self, l2_normalize=True):
        super(MaximumTransform, self).__init__(l2_normalize)

    def transform(self, x):
        return F.adaptive_max_pool2d(x, (1, 1))


@gin.configurable
class RMACTransform(FeatureTransform):
    def __init__(self, l2_normalize=True, config='1-3-5-G'):
        super(RMACTransform, self).__init__(l2_normalize)
        self.global_pool = False
        self.pooling_size = []
        self.strides = []

        for value in config.split('-'):
            if value == 'G':
                self.global_pool = True
            else:
                kernel_size = int(value)
                self.pooling_size.append(kernel_size)
                self.strides.append(1 if kernel_size == 1 else 2)

    def postprocess_macs(self, x):
        assert len(x.shape) == 4, f'Input shape is {x.shape}, but must be 4d'
        B, C, H, W = x.shape
        x = l2_normalize(x)
        if self.pca:
            x = l2_normalize(self.pca(to_bhwc(x).view(-1, C)))
            x = x.view(B, -1, C).transpose(1, 2).view(B, C, H, W)
        return x

    def transform(self, x):
        bs = x.size()[0]

        rmac = None
        for pooling_size, stride in zip(self.pooling_size, self.strides):
            pooled = torch.max_pool2d(x, pooling_size, stride)
            values = torch.sum(self.postprocess_macs(pooled), dim=(2, 3))
            if rmac is None:
                rmac = values
            else:
                rmac += values

        if self.global_pool:
            global_pool = F.adaptive_max_pool2d(x, (1, 1))
            rmac += self.postprocess_macs(global_pool).view(bs, -1)

        return rmac


class MACStackTransform(RMACTransform):
    """Transform that does not sum-pool the MACs but stacks them together

    This can be used to fit PCA to the individual MACs.
    """
    def __init__(self):
        super(RMACTransform, self).__init__(l2_normalize=False)

    def transform(self, x):
        channels = x.size()[1]
        macs = []

        for pooling_size, stride in zip(self.pooling_size, self.strides):
            pooled = l2_normalize(torch.max_pool2d(x, pooling_size, stride))
            macs.append(to_bhwc(pooled).view(-1, channels))

        if self.global_pool:
            global_pool = l2_normalize(F.adaptive_max_pool2d(x, (1, 1)))
            macs.append(to_bhwc(global_pool).view(-1, channels))

        return torch.cat(macs, dim=0)


def pca_from_sklearn(pca, whiten=True):
    return PCA(pca.components_, pca.mean_, pca.explained_variance_,
               whiten=whiten)


class PCA(nn.Module):
    def __init__(self, components, mean, explained_variance, whiten=False):
        super(PCA, self).__init__()
        self.whiten = whiten

        components = torch.as_tensor(components.T.astype(np.float32))
        mean = torch.as_tensor(mean.reshape(1, -1).astype(np.float32))
        explained_std = np.sqrt(explained_variance.reshape(1, -1))
        explained_std = torch.as_tensor(explained_std.astype(np.float32))

        self.register_buffer('components', components)
        self.register_buffer('mean', mean)
        self.register_buffer('explained_std', explained_std)

    def forward(self, x):
        assert len(x.shape) == 2, f'Input shape is {x.shape}, but must be 2d'
        x = x - self.mean
        x = torch.matmul(x, self.components)
        if self.whiten:
            x = x / self.explained_std

        return x
