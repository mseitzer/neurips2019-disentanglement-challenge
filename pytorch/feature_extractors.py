import collections
import os

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_save_name(model_name, model_id, interpolation_size):
    model_id = '_{}'.format(model_id) if model_id is not None else ''
    return '{}{}_{}x{}.pth'.format(model_name,
                                   model_id,
                                   interpolation_size[0],
                                   interpolation_size[1])


@gin.configurable
def get_feature_extractor(model_name,
                          interpolation_size=(224, 224),
                          stage=None,
                          pretrained=True,
                          custom_model_id=None):
    model_fns = {
        'VGG19': torchvision.models.vgg19_bn,
        'Inception': torchvision.models.inception_v3,
        'DenseNet-161': torchvision.models.densenet161,
        'ResNet-50': torchvision.models.resnet50,
        'ResNet-101': torchvision.models.resnet101,
        'ResNet-152': torchvision.models.resnet152,
        'ResNeXt-50': torchvision.models.resnext50_32x4d,
        'ResNeXt-101': torchvision.models.resnext101_32x8d
    }

    if model_name.startswith('Agg'):
        extractor = get_self_trained_agg_extractor(model_name,
                                                   interpolation_size)
        extractor.eval()
        return extractor

    if model_name not in model_fns:
        raise ValueError(f'Unknown feature extractor model {model_name}')

    pretrained = pretrained if custom_model_id is None else False
    model = model_fns[model_name](pretrained=pretrained)

    if custom_model_id is not None:
        model_name = '{}_{}'.format(model_name, custom_model_id)

    if model_name.startswith('ResNet') or model_name.startswith('ResNeXt'):
        extractor = ResNetExtractor(model, model_name, interpolation_size, stage)
    elif model_name.startswith('VGG'):
        extractor = VGGExtractor(model, model_name, interpolation_size)
    elif model_name.startswith('Dense'):
        extractor = DenseNetExtractor(model, model_name, interpolation_size)
    else:
        raise ValueError('Not implemented')

    if custom_model_id is not None:
        save_name = get_save_name(model_name, None, interpolation_size)
        save_path = os.path.join(os.getenv('TORCH_HOME'), save_name)
        extractor.load_state_dict(torch.load(save_path))

    extractor.eval()
    return extractor


def get_self_trained_agg_extractor(model_name, interpolation_size):
    assert model_name.startswith('Agg')
    base_extractor_name = model_name[3:]
    base_extractor = get_feature_extractor(base_extractor_name,
                                           interpolation_size,
                                           pretrained=False)
    extractor = AggregatedFeatureExtractor(base_extractor)

    save_path = os.path.join(os.getenv('TORCH_HOME'),
                             extractor.get_save_name())
    state_dict = torch.load(save_path, map_location=torch.device('cpu'))

    # Rename keys for old checkpoint compatibility
    if base_extractor_name == 'VGG19':
        state_dict_new = collections.OrderedDict()
        for key, value in state_dict.items():
            key = key.replace('model.', '')
            state_dict_new[key] = value
        state_dict = state_dict_new

    extractor.load_state_dict(state_dict)
    return extractor


def normalize_batch(tensor, mean, std):
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])


class Extractor(nn.Module):
    def __init__(self, model_name, interpolation_size, feature_dims):
        super(Extractor, self).__init__()
        self._model_name = model_name
        self._interpolation_size = interpolation_size
        self._feature_dims = feature_dims

    @property
    def model_name(self):
        return self._model_name

    @property
    def interpolation_size(self):
        return self._interpolation_size

    @property
    def feature_dims(self):
        return self._feature_dims


class ResNetExtractor(Extractor):
    def __init__(self, model, model_name, interpolation_size=(224, 224), stage=None):
        if not stage:
            feature_dims=2048
        else:
            feature_dims=int(128*np.min((2**stage, 2**4)))

        super(ResNetExtractor, self).__init__(model_name,
                                              interpolation_size,
                                              feature_dims=feature_dims)
        self.model = model
        self.stage = stage
        self.register_buffer('mean', torch.as_tensor(MEAN))
        self.register_buffer('std', torch.as_tensor(STD))

    def forward(self, x):
        # Check if we need interpolation. If so, interpolate.
        if self.interpolation_size != (64, 64):
            x = F.interpolate(x,
                              size=self.interpolation_size,
                              mode='bilinear',
                              align_corners=False)

        normalize_batch(x, self.mean, self.std)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # If stage is not none, then return output of respective stage.
        if self.stage:
            if self.stage >= 1:
                x = self.model.layer1(x)
            if self.stage >= 2:
                x = self.model.layer2(x)
            if self.stage >= 3:
                x = self.model.layer3(x)
            if self.stage >= 4:
                x = self.model.layer4(x)
        else:
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
        return x


class DenseNetExtractor(Extractor):
    def __init__(self, model, model_name, interpolation_size=(224, 224)):
        super(DenseNetExtractor, self).__init__(model_name,
                                                interpolation_size,
                                                feature_dims=2048)
        self.model = model
        self.register_buffer('mean', torch.as_tensor(MEAN))
        self.register_buffer('std', torch.as_tensor(STD))

    def forward(self, x):
        x = F.interpolate(x,
                          size=self.interpolation_size,
                          mode='bilinear',
                          align_corners=False)
        normalize_batch(x, self.mean, self.std)
        x = self.model.features(x)

        return x


class VGGExtractor(Extractor):
    def __init__(self, model, model_name, interpolation_size=(224, 224),
                 stage=None):
        if stage is not None:
            feature_dims = (64, 128, 256, 512, 512)[stage - 1]
        else:
            feature_dims = 512
        super(VGGExtractor, self).__init__(model_name,
                                           interpolation_size,
                                           feature_dims=feature_dims)
        if stage is not None:
            layers = []
            poolings = 0
            for layer in model.features:
                if isinstance(layer, nn.MaxPool2d):
                    poolings += 1
                if poolings == stage:
                    break
                layers.append(layer)
            self.features = nn.Sequential(*layers)
        else:
            self.features = model.features
        self.register_buffer('mean', torch.as_tensor(MEAN))
        self.register_buffer('std', torch.as_tensor(STD))

    def forward(self, x):
        x = F.interpolate(x,
                          size=self.interpolation_size,
                          mode='bilinear',
                          align_corners=False)
        normalize_batch(x, self.mean, self.std)

        return self.features(x)


@gin.configurable
class AggregatedFeatureExtractor(Extractor):
    """Extractor that learns to output one dimensional vectors"""
    def __init__(self, feature_extractor, features, kernel_sizes, strides, paddings,
                 dropout=0, l2_normalize=False, pooling=None, model_id=None):
        if len(features) == 0:
            output_features = feature_extractor.feature_dims
        else:
            output_features = features[-1]

        model_id = '_{}'.format(model_id) if model_id is not None else ''
        name = 'Agg' + feature_extractor.model_name + model_id
        super(AggregatedFeatureExtractor, self).__init__(name,
                                                         feature_extractor.interpolation_size,
                                                         output_features)
        self.l2_normalize = l2_normalize
        self.pooling = pooling
        self.extractor = feature_extractor

        layers = []
        if pooling == 'avg':
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif pooling == 'max':
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))

        input_dims = feature_extractor.feature_dims
        for num_features, kernel_size, stride, padding in zip(features, kernel_sizes, strides, paddings):
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            layers += [
                nn.Conv2d(input_dims, num_features, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU()
            ]
            input_dims = num_features
        self.aggregator = nn.Sequential(*layers)

    def get_save_name(self):
        return get_save_name(self.model_name, None, self.interpolation_size)

    def toggle_extractor(self, freeze):
        for param in self.extractor.parameters():
            param.requires_grad = not freeze

        for m in self.extractor.modules():
            if isinstance(m, nn.BatchNorm2d):
                if freeze:
                    # Set batch norm to eval mode such that running stats are
                    # not updated
                    m.eval()
                else:
                    m.train()

    def forward(self, x):
        feature_maps = self.extractor(x)
        features = self.aggregator(feature_maps)
        features = torch.reshape(features, (x.shape[0], -1))
        if self.l2_normalize:
            features = l2_normalize(features)
        return features


def l2_normalize(x):
    l2_norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
    # This is awkward, but ensures that this tensor lands on the correct
    # device during evaluation. The problem is that while tracing, a fixed
    # device is chosen when using `torch.ones_like(l2_norm)`, and this device
    # might be different from the evaluation device.
    ones = (l2_norm - l2_norm + 1.0)
    l2_norm = torch.where(l2_norm > 0.0, l2_norm, ones)
    return x / l2_norm
