import argparse
import math
import os
import sys
import time

import gin
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import utils_pytorch as pyu
from pytorch import feature_extractors


parser = argparse.ArgumentParser(description='Finetune extractor')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('config', help='Path to gin config')
parser.add_argument('experiment_name', help='Experiment name')

class_names = ['object_color',
               'object_shape',
               'object_size',
               'camera_height',
               'background_colors',
               'first_dof',
               'second_dof']


def filter_params(params):
    return filter(lambda p: p.requires_grad, params)


class Classifier(nn.Module):
    def __init__(self, feature_extractor, classes_per_category):
        super(Classifier, self).__init__()
        self.feature_extractor = feature_extractor

        self.heads = nn.ModuleList()
        for num_classes in classes_per_category:
            self.heads.append(nn.Linear(feature_extractor.feature_dims,
                                        num_classes))

    def loss(self, logits_per_category, targets):
        targets_per_category = targets.transpose(0, 1)

        losses = []
        for logits, target in zip(logits_per_category, targets_per_category):
            loss = F.cross_entropy(logits, target, reduction='mean')
            losses.append(loss)

        return losses

    def forward(self, x):
        features = self.feature_extractor(x)
        logits_per_category = []
        for head in self.heads:
            logits_per_category.append(head(features))

        return logits_per_category


def train_epoch(loader, model, optimizer, device, epoch_idx, log_interval,
                freeze_extractor=False):
    model.train()
    if freeze_extractor:
        model.feature_extractor.toggle_extractor(freeze=True)

    train_losses = {name: 0 for name in class_names}
    train_total_loss = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()

        logits = model(data)
        losses = model.loss(logits, targets)

        total_loss = sum(losses)
        total_loss.backward()

        optimizer.step()

        train_total_loss += total_loss.item() * len(data)
        for name, loss in zip(class_names, losses):
            train_losses[name] += loss.item() * len(data)

        if batch_idx % log_interval == 0:
            s = ('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                 .format(epoch_idx,
                         batch_idx * len(data),
                         len(loader.dataset),
                         100. * batch_idx / len(loader)))
            s += ', '.join('{}: {:.4f}'.format(name, loss.item())
                           for name, loss in zip(class_names, losses))
            print(s)

    s = ('====> Epoch: {} Avg. total loss: {:.4f}, '
         .format(epoch_idx, train_total_loss / len(loader.dataset)))
    s += ', '.join('{} loss: {:.4f}'.format(name, loss / len(loader.dataset))
                   for name, loss in train_losses.items())
    print(s)


def val_epoch(loader, model, device):
    model.eval()

    test_losses = {name: 0 for name in class_names}
    test_correct = {name: 0 for name in class_names}
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data, targets = data.to(device), targets.to(device)

            logits_per_category = model(data)
            losses = model.loss(logits_per_category, targets)

            for name, logits, target in zip(class_names,
                                            logits_per_category,
                                            targets.transpose(0, 1)):
                pred = logits.argmax(dim=1, keepdim=True)
                test_correct[name] += pred.eq(target.view_as(pred)).sum().item()

            for name, loss in zip(class_names, losses):
                test_losses[name] += loss.item() * len(data)

    s = 'Test Losses: \t'
    s += ', '.join('{}: {:.4f}'.format(name, loss / len(loader.dataset))
                   for name, loss in test_losses.items())
    print(s)
    s = 'Test Accuracies: \t'
    s += ', '.join('{}: {:.4f}'.format(name, correct / len(loader.dataset))
                   for name, correct in test_correct.items())
    print(s)


@gin.configurable
def train(train_loader, val_loader, model, device, experiment_name,
          epochs=20, optimizer_name='radam', lr=1e-3,
          weight_decay=0,
          unfreeze_extractor_epoch=0,
          extractor_lr=1e-4,
          log_interval=50):
    freeze_extractor = True
    model.feature_extractor.toggle_extractor(freeze=True)

    parameter_groups = [
        {'params': filter_params(model.parameters()), 'lr': lr},
    ]

    if optimizer_name == 'adam':
        optimizer = optim.Adam(parameter_groups, lr=lr, betas=(0.9, 0.999),
                               weight_decay=weight_decay)
    elif optimizer_name == 'radam':
        from radam import RAdam
        optimizer = RAdam(parameter_groups, lr=lr, betas=(0.9, 0.999),
                          weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(parameter_groups, lr=lr,
                                  weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer {optimizer_name}')

    for epoch in range(1, epochs + 1):
        print('Learning rate is {}'.format(optimizer.param_groups[0]['lr']))

        if epoch - 1 == unfreeze_extractor_epoch:
            print('Unfreezing extractor')
            freeze_extractor = False
            model.feature_extractor.toggle_extractor(freeze=False)
            pgroup = {'params': filter_params(model.feature_extractor.extractor.parameters()),
                      'lr': extractor_lr}
            optimizer.add_param_group(pgroup)

        start_time = time.time()
        train_epoch(train_loader, model, optimizer, device, epoch, log_interval,
                    freeze_extractor=freeze_extractor)
        end_time = time.time()
        print(f'Epoch took {end_time-start_time:.2f} seconds')

        val_epoch(val_loader, model, device)

        save_model(model, experiment_name)

    return model


def save_model(model, experiment_name):
    base_path = os.getenv('AICROWD_OUTPUT_PATH', './scratch/shared')
    save_dir = os.path.join(base_path, experiment_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(model.feature_extractor.state_dict(),
               os.path.join(save_dir, model.feature_extractor.get_save_name()))


@gin.configurable
def main(args,
         seed=1,
         dataset_name='mpi3d_realistic',
         dataset_size=0,
         batch_size=128,
         test_ratio=0.2):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    print(f'Running on {device}')

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not isinstance(dataset_name, list):
        dataset_name = [dataset_name]

    train_datasets = []
    val_datasets = []
    for name in dataset_name:
        dataset = pyu.DLIBDatasetWithClasses(name)

        if dataset_size <= 0:
            dataset_size = len(dataset)
        else:
            dataset.iterator_len = dataset_size

        val_samples = math.floor(len(dataset) * test_ratio)
        train_samples = len(dataset) - val_samples
        train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                                   (train_samples, val_samples))
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    if len(train_datasets) > 1:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    extractor = feature_extractors.get_feature_extractor()
    agg_extractor = feature_extractors.AggregatedFeatureExtractor(extractor)
    classifier = Classifier(agg_extractor, dataset.dataset.factors_num_values)

    classifier.to(device)

    train(train_loader, val_loader, classifier, device, args.experiment_name)


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    gin.parse_config_file(args.config)
    main(args)
