import math

import gin
import torch
from torch import nn


@gin.configurable
class RegularizationLoss(nn.Module):
    def __init__(self,
                 latent_dims,
                 scale_by_batch=True,
                 use_bayes_factor_vae0_loss=False,
                 use_tc_loss=False):
        super(RegularizationLoss, self).__init__()
        self.scale_by_batch = scale_by_batch
        self.use_bayes_factor_vae0_loss = use_bayes_factor_vae0_loss
        self.use_tc_loss = use_tc_loss

        if use_bayes_factor_vae0_loss:
            self.log_precision = nn.Parameter(torch.zeros(1, latent_dims))

    def add_kld_loss(self, losses, mu, logvar):
        """Standard KLD with standard Gaussian as prior

        Computes `0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)`

        See Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        """
        x = 1 + logvar - mu.pow(2) - logvar.exp()
        KLD = -0.5 * torch.sum(x)

        losses['KLD'] = KLD / mu.shape[-1]

    def add_bayes_factor_vae0_loss(self, losses, mu, logvar):
        """KLD with Gaussian with flexible variances as prior

        The target precision (reciprocal of variance) of the prior can be
        learned from data. Then we can compute the KLD as

        `0.5 * sum(1 + log(sigma^2) + log(alpha) - mu^2 * alpha -
                   sigma^2 * alpha)`

        where alpha is the learned precision parameter to be learned
        from data. Formula is self-derived and thus may contain errors.

        See model BF-VAE-0 from Kim et al. Bayes-Factor-VAE, 2019,
        https://arxiv.org/abs/1909.02820
        """
        x = (1 + logvar + self.log_precision
             - mu.pow(2) * self.log_precision.exp()
             - logvar.exp() * self.log_precision.exp())

        KLD = -0.5 * torch.sum(x)
        losses['KLD'] = KLD / mu.shape[-1]

        # Compute penalty term that specifies that variance should be close
        # to one
        alpha_penalty = torch.sum((1 / self.log_precision.exp() - 1).pow(2))
        losses['alpha_penalty'] = alpha_penalty / mu.shape[-1]

    def add_tc_loss(self, losses, z, mu, logvar):
        """Total correlation loss

        Computes `KL[q(z) || prod_i z_i]`

        Adapted from
        https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py
        under MIT License

        See Chen et al. Isolating Sources of Disentanglement in VAEs, 2018,
        https://arxiv.org/abs/1802.04942
        """
        mat_log_qz = _matrix_log_density_gaussian(z, mu, logvar)

        log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
        log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

        tc_loss = torch.sum(log_qz - log_prod_qzi)
        losses['TC'] = tc_loss / mu.shape[1]

    def forward(self, z, mu, logvar):
        losses = {}

        if self.use_bayes_factor_vae0_loss:
            self.add_bayes_factor_vae0_loss(losses, mu, logvar)
        else:
            self.add_kld_loss(losses, mu, logvar)

        if self.use_tc_loss:
            self.add_tc_loss(losses, z, mu, logvar)

        if self.scale_by_batch:
            for name, loss in losses.items():
                losses[name] = loss / mu.shape[0]

        return losses


def _matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of batch pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Adapted from
    https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py
    under MIT License

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).
    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).
    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return _log_density_gaussian(x, mu, logvar)


def _log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian

    Adapted from
    https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py
    under MIT License

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.
    mu: torch.Tensor or np.ndarray or float
        Mean.
    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density
