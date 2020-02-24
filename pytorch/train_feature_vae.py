__doc__ = """
Example training script with PyTorch. Here's what you need to do.

Before you run this script, ensure that the following environment variables are set:
    1. AICROWD_OUTPUT_PATH (default: './scratch/shared')
    2. AICROWD_EVALUATION_NAME (default: 'experiment_name')
    3. AICROWD_DATASET_NAME (default: 'cars3d')
    4. DISENTANGLEMENT_LIB_DATA (you may set this to './scratch/dataset' if that's
                                 where the data lives)

We provide utility functions to make the data and model logistics painless.
But this assumes that you have set the above variables correctly.

Once you're done with training, you'll need to export the function that returns
the representations (which we evaluate). This function should take as an input a batch of
images (NCHW) and return a batch of vectors (NC), where N is the batch-size, C is the
number of channels, H and W are height and width respectively.

To help you with that, we provide an `export_model` function in utils_pytorch.py. If your
representation function is a torch.jit.ScriptModule, you're all set
(just call `export_model(model)`); if not, it will be traced (!) and the resulting ScriptModule
will be written out. To learn what tracing entails:
https://pytorch.org/docs/stable/jit.html#torch.jit.trace

You'll find a few more utility functions in utils_pytorch.py for pytorch related stuff and
for data logistics.
"""
import argparse
import math
import os
import sys
import time

import gin
import dill
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn import decomposition

import aicrowd_helpers
import utils_pytorch as pyu
from pytorch import feature_extractors, feature_transforms, vae


parser = argparse.ArgumentParser(description='VAE training script')
parser.add_argument('--features-dir', default='./scratch/features',
                    help='Directory where stored features reside')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('config', help='Path to gin config')
parser.add_argument('experiment_name', help='Experiment name')


def write_config(experiment_name):
    """Write output config"""
    base_path = os.getenv('AICROWD_OUTPUT_PATH', '../scratch/shared')
    path = os.path.join(base_path, experiment_name)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, 'conf.gin'), 'w') as f:
        f.write(gin.config_str())


class StatsLogger:
    def __init__(self, experiment_name):
        base_path = os.getenv('AICROWD_OUTPUT_PATH', '../scratch/shared')
        path = os.path.join(base_path, experiment_name)
        if not os.path.exists(path):
            os.mkdir(path)

        self.path = path
        self.dump_header = True

    def append(self, epoch, stats):
        with open(os.path.join(self.path, 'stats.csv'), 'a') as f:
            if self.dump_header:
                keys = ['epochs'] + sorted(list(stats))
                f.write(';'.join(keys) + '\n')
                self.dump_header = False
            values = [str(epoch)] + [str(stats[key]) for key in sorted(stats)]
            f.write(';'.join(values) + '\n')


class RepresentationExtractor(nn.Module):
    """Extract representations (latent factors) from images

    This is what gets saved on disk and loaded for the evaluation in the end
    """
    def __init__(self, feature_extractor, feature_transform, vae_extractor):
        super(RepresentationExtractor, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_transform = feature_transform
        self.vae_extractor = vae_extractor

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.feature_transform(features)
        return self.vae_extractor(features)


class FeatureDataset(Dataset):
    """Dataset over a numpy array of features"""
    def __init__(self, features):
        self.features = features

    @property
    def dims(self):
        if len(self.features.shape) == 2:
            return self.features.shape[-1]
        else:
            return self.features.shape[1:]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return torch.from_numpy(self.features[item])


def extract_features(extractor, loader, device, feature_transform=None):
    """Iterable that extracts features from images and optionally transforms them"""
    for batch_idx, data in enumerate(loader):
        with torch.no_grad():
            data = data.to(device).float()
            features = extractor(data)
            if feature_transform:
                features = feature_transform(features)
            yield features


def transform_features(feature_transform, loader, device):
    """Iterable that transforms features"""
    for batch_idx, features in enumerate(loader):
        with torch.no_grad():
            features = features.to(device).float()
            features = feature_transform(features)
            yield features


def get_and_dump_features(extractor, loader, device, path,
                          dataset_size, log_interval):
    """Convenience function extracting a feature array and saving it to disk"""
    extract_iter = extract_features(extractor, loader, device)
    feature_array = get_feature_array(extract_iter,
                                      dataset_size,
                                      len(loader),
                                      log_interval,
                                      path)
    return feature_array


def get_feature_array(extract_iter, dataset_size, num_batches, log_interval,
                      file_path=None):
    """Function that batchwise fills a numpy array from an iterable

    If `file_path` is set, directly saves the array to disk and returns a
    memory-mapped array.
    """
    def init_array(feature_shape):
        shape = (dataset_size,) + feature_shape
        if file_path:
            array = np.lib.format.open_memmap(file_path,
                                              mode='w+',
                                              dtype=np.float32,
                                              shape=shape)
        else:
            array = np.zeros(shape, dtype=np.float32)

        return array

    idx = 0
    feature_array = None
    for batch_idx, features in enumerate(extract_iter):
        features = features.cpu().numpy()
        if feature_array is None:
            # Lazy init because feature shape is only fully known now
            feature_array = init_array(features.shape[1:])

        feature_array[idx:idx + len(features)] = features
        idx += len(features)
        if batch_idx % log_interval == 0:
            print('Extracted batch {}/{}'.format(batch_idx, num_batches))

    return feature_array


def fit_pca_to_macs(loader, device, log_interval, whiten):
    """Extract MACs and use them to fit a sklearn PCA"""
    print('Fitting PCA to whiten MACs...')
    feature_transform = feature_transforms.MACStackTransform()
    pca = None
    for idx, features in enumerate(transform_features(feature_transform,
                                                      loader,
                                                      device)):
        features = features.cpu().numpy()
        if pca is None:
            # Lazy init
            num_features = features.shape[1]
            pca = decomposition.IncrementalPCA(n_components=num_features,
                                               whiten=whiten)
        pca.partial_fit(features)
        if idx % log_interval == 0:
            print('Fitted batch {}/{}'.format(idx, len(loader)))

    return pca


def get_and_fit_pca(features_dir, model_name, dataset_size, interpolation_size,
                    feature_loader, device, log_interval, whiten):
    """Convenience function that either fits PCA or loads it from disk"""
    dataset_name = pyu.get_dataset_name()
    pca_path = (f'{features_dir}/pca_{dataset_name}_{model_name}'
                f'_{size[0]}x{size[1]}_{dataset_size}.dill')
    if not os.path.exists(pca_path):
        pca = fit_pca_to_macs(feature_loader, device, log_interval,
                              whiten=True)
        with open(pca_path, 'wb') as f:
            dill.dump(pca, f, protocol=dill.HIGHEST_PROTOCOL)
    else:
        print(f'Reusing PCA from {pca_path}')
        with open(pca_path, 'rb') as f:
            pca = dill.load(f)
    return feature_transforms.pca_from_sklearn(pca, whiten=whiten)


def get_transformed_feature_dataset(feature_extractor,
                                    feature_transform,
                                    loader,
                                    dataset_size,
                                    cache_features_on_disk,
                                    features_dir,
                                    use_pca,
                                    device,
                                    log_interval):
    """Extract and transform features"""

    # Load features from disk or extract and store them
    feature_dataset = None
    if cache_features_on_disk:
        print('Extracting features...')
        dataset_name = pyu.get_dataset_name()
        size = feature_extractor.interpolation_size
        features_path = (f'{features_dir}/{dataset_name}_{feature_extractor.model_name}'
                         f'_{size[0]}x{size[1]}_{dataset_size}.npy')
        if not os.path.exists(features_path):
            print(f'Dumping features to {features_path}')
            feature_array = get_and_dump_features(feature_extractor,
                                                  loader,
                                                  device,
                                                  features_path,
                                                  dataset_size,
                                                  log_interval)
        else:
            print(f'Reusing features from {features_path}')
            feature_array = np.lib.format.open_memmap(features_path,
                                                      mode='r')
            if dataset_size < len(feature_array):
                feature_array = feature_array[:dataset_size]

        feature_dataset = FeatureDataset(feature_array)

    # Maybe fit a PCA to feature maps
    pca = None
    if use_pca:
        # PCA is fitted to each individual entry in the feature maps, not the
        # aggregated feature vector (although that might make sense as well).
        if not cache_features_on_disk:
            extract_iter = extract_features(feature_extractor, loader, device)
            feature_array = get_feature_array(extract_iter,
                                              dataset_size,
                                              len(loader),
                                              log_interval)
            feature_dataset = FeatureDataset(feature_array)
        feature_loader = DataLoader(feature_dataset,
                                    batch_size=256,
                                    shuffle=False,
                                    num_workers=2,
                                    pin_memory=True)

        pca = get_and_fit_pca(features_dir, feature_extractor.model_name,
                              dataset_size, feature_extractor.interpolation_size,
                              feature_loader, device, log_interval,
                              whiten=False)
        feature_transform.pca = pca
        feature_transform = feature_transform.to(device)

    if cache_features_on_disk:
        print('Transforming features...')
        feature_loader = DataLoader(feature_dataset,
                                    batch_size=256,
                                    shuffle=False,
                                    num_workers=2,
                                    pin_memory=True)
        num_batches = len(feature_loader)
        extract_iter = transform_features(feature_transform,
                                          feature_loader,
                                          device)
    else:
        # If we do not have features on disk, we are doing joint extraction
        # and transformation for some efficiency
        print('Extracting and transforming features...')
        num_batches = len(loader)
        extract_iter = extract_features(feature_extractor,
                                        loader,
                                        device,
                                        feature_transform)

    feature_array = get_feature_array(extract_iter,
                                      dataset_size,
                                      num_batches,
                                      args.log_interval)

    return FeatureDataset(feature_array)


@gin.configurable
class BetaSchedule:
    """Class controlling the schedule of beta"""
    def __init__(self,
                 schedule_type='linear',
                 beta0=0.001,
                 beta1=0.1,
                 beta_start=10,
                 beta_end=100,
                 cycles=1):
        """
        :param schedule_type:
            - `linear`: Linear increase from `beta0` to `beta1`
            - `cosine`: Cosine increase from `beta0` to `beta1`
            - `cosine-restarts`: Cosine with several restarts
        :param beta0: Initial value of beta
        :param beta1: Final value of beta
        :param beta_start: Index of epoch where schedule begins
        :param beta_end: Index of epoch where schedule ends
        :param cycles: Number of cycles to use for `cosine-restarts`
        """
        assert schedule_type in ('constant', 'linear',
                                 'cosine', 'cosine-restarts')
        assert beta_start <= beta_end
        if schedule_type == 'constant':
            assert beta0 == beta1, 'Need beta0==beta1 for schedule `constant`'

        self.schedule_type = schedule_type
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cycles = cycles

    def get_beta(self, epoch, beta_start=None, beta_end=None):
        """Get current value of beta for epoch

        Note that the epoch value is zero-indexed
        """
        if beta_start is None:
            beta_start = self.beta_start
        if beta_end is None:
            beta_end = self.beta_end

        if epoch < beta_start:
            return self.beta0
        elif beta_start <= epoch <= beta_end:
            if self.schedule_type == 'constant':
                return self.beta0
            elif self.schedule_type == 'linear':
                return (self.beta0
                        + (self.beta1 - self.beta0) * (epoch - beta_start)
                        / (beta_end - beta_start))
            elif self.schedule_type == 'cosine':
                return self.cosine_annealing(epoch, beta_start, beta_end)
            elif self.schedule_type == 'cosine-restarts':
                cycle_len = (beta_end - beta_start) // self.cycles
                cycle_idx = (epoch - beta_start) // cycle_len
                start_epoch = beta_start + cycle_len * cycle_idx
                end_epoch = beta_start + cycle_len * (cycle_idx + 1)
                return self.cosine_annealing(epoch, start_epoch, end_epoch)
            else:
                raise ValueError('Unsupported schedule type {}'
                                 .format(self.schedule_type))
        else:
            return self.beta1

    def cosine_annealing(self, epoch, start_epoch, end_epoch):
        return (self.beta1 - 0.5 * (self.beta1 - self.beta0)
                * (1 + math.cos(math.pi * (epoch - start_epoch)
                                / (end_epoch - start_epoch))))


@gin.configurable
class LRSchedule:
    """Simple learning rate schedule for warmup"""
    def __init__(self, lr, warmup_lr=None, warmup_epochs=0):
        self.lr = lr
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch):
        if self.warmup_lr is not None and epoch < self.warmup_epochs:
            return self.warmup_lr
        else:
            return self.lr


def train_epoch(loader, vae, optimizer, device, epoch_idx, log_interval,
                loss_weights, stats_logger, clip_gradients=None):
    """Train VAE for an epoch"""
    vae.train()

    train_losses = {}
    train_total_loss = 0
    for batch_idx, data in enumerate(loader):
        data = data.to(device).float()
        target = data

        optimizer.zero_grad()

        decoder_output, z, mu, logvar = vae(data)

        losses = vae.loss(decoder_output, target, z, mu, logvar)

        total_loss = sum(loss_weights.get(loss_name, 1) * loss
                         for loss_name, loss in losses.items()
                         if '_unweighted' not in loss_name)
        total_loss.backward()

        if clip_gradients is not None:
            torch.nn.utils.clip_grad_value_(vae.parameters(), clip_gradients)

        optimizer.step()

        train_total_loss += total_loss.item() * len(data)

        for name, loss in losses.items():
            train_loss = train_losses.setdefault(name, 0)
            train_losses[name] = train_loss + loss.item() * len(data)

        if batch_idx % log_interval == 0:
            s = ('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                 .format(epoch_idx,
                         batch_idx * len(data),
                         len(loader.dataset),
                         100. * batch_idx / len(loader)))
            s += ', '.join('Loss {}: {:.7f}'.format(name, loss.item())
                           for name, loss in losses.items())
            print(s)

    stats = {name: loss / len(loader.dataset)
             for name, loss in train_losses.items()}
    stats['total_loss'] = train_total_loss / len(loader.dataset)

    s = ('====> Epoch: {} Avg. total loss: {:.7f}, '
         .format(epoch_idx, stats['total_loss']))
    s += ', '.join('{} loss: {:.7f}'.format(name, loss)
                   for name, loss in stats.items() if name != 'total_loss')
    print(s)

    # Add weighted losses for logging
    for name, loss in train_losses.items():
        weight = loss_weights.get(name, 1)
        stats['weighted_' + name] = weight * loss / len(loader.dataset)

    return stats


@gin.configurable
def train_vae(loader,
              device,
              stats_logger,
              lr=1e-3,
              schedule_lr=False,
              latent_dims=16,
              epochs=100,
              optimizer_name='adam',
              adam_beta1=0.5,
              loss_weights=None,
              extractor_lr=1e-5,
              clip_gradients=None,
              encoder_class=vae.Encoder,
              decoder_class=vae.Decoder,
              schedule_classes=None,
              beta_schedule_class=BetaSchedule):
    """Entry point for VAE training"""
    repr_dims = loader.dataset.dims

    encoder = encoder_class(repr_dims, latent_dims)
    decoder = decoder_class(repr_dims, latent_dims)
    model = vae.VAE(encoder, decoder).to(device)

    if schedule_lr:
        # LambdaLR multiplies the initial learning rate with the value
        # returned from lambda each epoch. If we want to directly use the
        # value returned from lambda as the learning rate, we can set an
        # initial learning rate of 1.
        initial_lr = lr
        lr = 1.0

    parameter_groups = [
        {'params': model.parameters(), 'lr': lr},
    ]

    if optimizer_name == 'adam':
        optimizer = optim.Adam(parameter_groups,
                               lr=lr,
                               betas=(adam_beta1, 0.999))
    elif optimizer_name == 'radam':
        from radam import RAdam
        optimizer = RAdam(parameter_groups,
                          lr=lr,
                          betas=(adam_beta1, 0.999))
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(parameter_groups,
                                  lr=lr)
    else:
        raise ValueError(f'Unknown optimizer {optimizer_name}')

    if schedule_lr:
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                   LRSchedule(initial_lr))

    if loss_weights is None:
        loss_weights = {}
    else:
        assert isinstance(loss_weights, dict), \
            'Loss weights must be a dictionary `loss_name -> weight`'

    if schedule_classes is None:
        schedule_classes = {}
    else:
        assert isinstance(schedule_classes, dict), \
            'schedules_classes must be a dictionary `loss_name -> schedule_class`'
    schedule_classes['KLD'] = beta_schedule_class

    loss_schedules = {name: schedule_class()
                      for name, schedule_class in schedule_classes.items()}

    print('Training VAE on features...')
    for epoch in range(1, epochs + 1):
        print('Learning rate is {}'.format(optimizer.param_groups[0]['lr']))
        for name, schedule in loss_schedules.items():
            if name == 'KLD':
                # Special case for KLD's weight (beta)
                if isinstance(schedule, BetaSchedule):
                    beta = schedule.get_beta(epoch - 1)
                    loss_weights['KLD'] = beta
                    print(f'Beta is {beta}')
            else:
                loss_weights[name] = schedule.get_beta(epoch - 1)

        if model.reg_loss.use_bayes_factor_vae0_loss:
            variances = (1 / model.reg_loss.log_precision.exp()).cpu().detach().numpy()
            print(variances[variances > 1])

        start_time = time.time()
        epoch_stats = train_epoch(loader,
                                  model,
                                  optimizer,
                                  device,
                                  epoch,
                                  1,
                                  loss_weights,
                                  stats_logger,
                                  clip_gradients)
        end_time = time.time()
        print(f'Epoch took {end_time-start_time:.2f} seconds')
        stats_logger.append(epoch - 1, epoch_stats)

        if schedule_lr:
            lr_scheduler.step()

    return model


@gin.configurable
def main(args,
         seed=1,
         dataset_size=0,
         vae_batch_size=64,
         use_pca=False,
         cache_features_on_disk=True,
         sequential_dataset=False,
         transform_class=feature_transforms.MaximumTransform):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    write_config(args.experiment_name)
    stats_logger = StatsLogger(args.experiment_name)

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f'Running on {device}')

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    shuffle = False if sequential_dataset else True
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    loader = pyu.get_loader(batch_size=args.batch_size,
                            sequential=sequential_dataset,
                            shuffle=shuffle,
                            **kwargs)

    if sequential_dataset:
        if dataset_size <= 0:
            dataset_size = len(loader.dataset)
        else:
            loader.dataset.iterator_len = dataset_size
    else:
        if dataset_size <= 0:
            loader.dataset.iterator_len = len(loader.dataset.dataset.images)
            dataset_size = loader.dataset.iterator_len
        else:
            loader.dataset.iterator_len = dataset_size

    print(f'Training with {loader.dataset.iterator_len} images')

    aicrowd_helpers.execution_start()
    aicrowd_helpers.register_progress(0.)

    feature_extractor = feature_extractors.get_feature_extractor()
    feature_extractor.to(device)
    feature_transform = transform_class()
    feature_transform.to(device)

    # Extract and transform features from images
    dataset = get_transformed_feature_dataset(feature_extractor,
                                              feature_transform,
                                              loader,
                                              dataset_size,
                                              cache_features_on_disk,
                                              args.features_dir,
                                              use_pca,
                                              device,
                                              args.log_interval)
    # Train VAE on the aggregated features
    loader = DataLoader(dataset,
                        batch_size=vae_batch_size,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True)
    aicrowd_helpers.register_progress(0.40)

    model = train_vae(loader, device, stats_logger)

    aicrowd_helpers.register_progress(0.90)

    # Export the representation extractor
    vae_extractor = vae.get_representation_extractor(model)
    pyu.export_model(RepresentationExtractor(feature_extractor,
                                             feature_transform,
                                             vae_extractor),
                     input_shape=(1, 3, 64, 64),
                     cuda=args.cuda)

    # Done!
    aicrowd_helpers.register_progress(1.0)
    aicrowd_helpers.submit()


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    gin.parse_config_file(args.config)
    main(args)
