import gin
import torch
from torch import nn
from torch.nn import functional as F


class Likelihood(nn.Module):
    def __init__(self, loss_name, input_dims, output_dims,
                 scale_loss_by_batch=True,
                 scale_loss_by_dims=False):
        super(Likelihood, self).__init__()
        self._loss_name = loss_name
        self.scale_loss_by_batch = scale_loss_by_batch
        self.scale_loss_by_dims = scale_loss_by_dims

    @property
    def loss_name(self):
        return self._loss_name

    def scale_and_wrap_loss(self, loss, batch_size, dim_size):
        if self.scale_loss_by_batch:
            loss = loss / batch_size
        if self.scale_loss_by_dims:
            loss = loss / dim_size

        return {self.loss_name: loss}

    def loss(self, dist_parameters, x):
        raise NotImplementedError('Must be implemented in child classes')

    def forward(self, x):
        raise NotImplementedError('Must be implemented in child classes')


@gin.configurable
class GaussianLikelihood(Likelihood):
    def __init__(self, input_dims, output_dims,
                 scale_mean_zero_one=True,
                 scale_loss_by_batch=True,
                 scale_loss_by_dims=False):
        super(GaussianLikelihood, self).__init__('MSE',
                                                 input_dims,
                                                 output_dims,
                                                 scale_loss_by_batch,
                                                 scale_loss_by_dims)
        self.scale_mean_zero_one = scale_mean_zero_one
        self.mean = nn.Linear(input_dims, output_dims)

    def loss(self, dist_parameters, x):
        mean = dist_parameters
        loss = F.mse_loss(mean, x, reduction='sum')
        return self.scale_and_wrap_loss(loss, x.shape[0], x.shape[1])

    def forward(self, x):
        mean = self.mean(x)
        if self.scale_mean_zero_one:
            mean = torch.sigmoid(mean)
        return mean


@gin.configurable
class GaussianLikelihood2d(Likelihood):
    def __init__(self, input_dims, output_dims,
                 scale_mean_zero_one=True,
                 scale_loss_by_batch=True,
                 scale_loss_by_dims=False):
        super(GaussianLikelihood2d, self).__init__('MSE',
                                                   input_dims,
                                                   output_dims,
                                                   scale_loss_by_batch,
                                                   scale_loss_by_dims)
        self.scale_mean_zero_one = scale_mean_zero_one
        self.mean = nn.Conv2d(input_dims, output_dims, 1)

    def loss(self, dist_parameters, x):
        mean = dist_parameters
        loss = F.mse_loss(mean, x, reduction='sum')
        return self.scale_and_wrap_loss(loss,
                                        x.shape[0],
                                        x.shape[1] * x.shape[2] * x.shape[3])

    def forward(self, x):
        mean = self.mean(x)
        if self.scale_mean_zero_one:
            mean = torch.sigmoid(mean)
        return mean
