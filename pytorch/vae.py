import gin
import torch
from torch import nn

import likelihoods
import reg_losses


@gin.configurable
def get_representation_extractor(vae, pca=None, top_n_variance_dims=None):
    dimension_indices = None
    if top_n_variance_dims is not None and vae.reg_loss.use_bayes_factor_vae0_loss:
        # Restrict representation to the N dimensions with the lowest
        # precision, i.e. the largest variance.
        # This only works for the KLD loss variants learning the variance
        # of the prior from data
        _, dimension_indices = torch.topk(vae.reg_loss.log_precision,
                                          top_n_variance_dims,
                                          dim=1,
                                          largest=False)
        dimension_indices = dimension_indices.squeeze()

    return RepresentationExtractor(vae.encoder, dimension_indices=dimension_indices)


class RepresentationExtractor(nn.Module):
    VALID_MODES = ['mean', 'sample']

    def __init__(self, encoder, mode='mean', dimension_indices=None):
        super(RepresentationExtractor, self).__init__()
        assert mode in self.VALID_MODES, f'`mode` must be one of {self.VALID_MODES}'
        self.encoder = encoder
        self.mode = mode
        self.dimension_indices = dimension_indices

    def forward(self, x):
        mu, logvar = self.encoder(x)
        if self.mode == 'mean':
            z = mu
        elif self.mode == 'sample':
            z = self.reparameterize(mu, logvar)
        else:
            raise NotImplementedError

        if self.dimension_indices is not None:
            z = torch.index_select(z, 1, self.dimension_indices)

        return z

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


@gin.configurable
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reg_loss = reg_losses.RegularizationLoss(encoder.latent_dims)

    def loss(self, likelihood_parameters, x, z, mu, logvar):
        losses = {}
        losses.update(self.decoder.output_dist.loss(likelihood_parameters, x))
        losses.update(self.reg_loss(z, mu, logvar))
        return losses

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = RepresentationExtractor.reparameterize(mu, logvar)
        return self.decoder(z), z, mu, logvar


@gin.configurable
class Encoder(nn.Module):
    def __init__(self, input_dims, latent_dims, features_per_layer,
                 batch_norm=False, dropout=0, init='default'):
        super(Encoder, self).__init__()
        self.latent_dims = latent_dims
        use_bias = not batch_norm

        layers = []
        out_dims = input_dims
        for num_features in features_per_layer:
            layers.append(nn.Linear(out_dims, num_features, bias=use_bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            out_dims = num_features

        self.tail = nn.Sequential(*layers)
        self.head_mu = nn.Linear(out_dims, latent_dims)
        self.head_logvar = nn.Linear(out_dims, latent_dims)
        self.apply(lambda layer: init_layer(layer, init))

    def forward(self, x):
        h = self.tail(x)
        return self.head_mu(h), self.head_logvar(h)


@gin.configurable
class Decoder(nn.Module):
    def __init__(self, output_dims, latent_dims, features_per_layer,
                 batch_norm=False, dropout=0, init='default',
                 likelihood_class=likelihoods.GaussianLikelihood):
        super(Decoder, self).__init__()
        use_bias = not batch_norm

        layers = []
        out_dims = latent_dims
        for num_features in features_per_layer:
            layers.append(nn.Linear(out_dims, num_features, bias=use_bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            out_dims = num_features
        self.layers = nn.Sequential(*layers)
        self.output_dist = likelihood_class(input_dims=out_dims,
                                            output_dims=output_dims)
        self.apply(lambda layer: init_layer(layer, init))

    def forward(self, x):
        x = self.layers(x)
        return self.output_dist(x)


def init_layer(layer, init_type):
    assert init_type in ('orthogonal', 'default')
    if isinstance(layer, nn.Linear):
        if init_type == 'orthogonal':
            nn.init.orthogonal_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.BatchNorm1d):
        torch.nn.init.ones_(layer.weight)
