from collections import OrderedDict
import numpy as np
from PIL import Image
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from .modules import RavelTensor, SobelFilter

# __all__ = ['AlexNet', 'AlexNetSobel', 'AlexNetTruncated']


class AlexNet(nn.Module):
    """
    Modified Alexnet with BatchNorm after every conv and FC layer except the last FC layer.

    conv-batchnorm-relu
    fc-batchnorm-relu
    """

    # index of the layer in self.features
    feature_layer_index = {
        'conv1': 0,
        'conv2': 4,
        'conv3': 8,
        'conv4': 11,
        'conv5': 14,
        'pool5': 17
    }

    def __init__(self, input_channels=3, num_classes=1000, batch_norm_momentum=0.01):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.is_sobel = False
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),  # conv1 = 0
            nn.BatchNorm2d(96, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # conv2 = 4
            nn.BatchNorm2d(256, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # conv3 = 8
            nn.BatchNorm2d(384, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # conv4 = 11
            nn.BatchNorm2d(384, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # conv5 = 14
            nn.BatchNorm2d(256, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(OrderedDict([
            ('fc6', nn.Linear(256 * 6 * 6, 4096)),
            ('bn6', nn.BatchNorm1d(4096, momentum=batch_norm_momentum)),
            ('relu6', nn.ReLU(inplace=True)),
            ('dropout6', nn.Dropout()),
            ('fc7', nn.Linear(4096, 4096)),
            ('bn7', nn.BatchNorm1d(4096, momentum=batch_norm_momentum)),
            ('relu7', nn.ReLU(inplace=True)),
            ('dropout7', nn.Dropout()),  # don't use this one ? Training from noise didn't use
            ('fc8', nn.Linear(4096, num_classes))])
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if name != 'classifier.fc8':
                    nn.init.normal_(m.weight, 0, 0.005)
                    nn.init.constant_(m.bias, 0.1)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetSobel(nn.Module):
    """
    Alexnet with fixed Sobel filter as the 0-th layer.
    """

    def __init__(self, num_classes=1000, batch_norm_momentum=0.01):
        super(AlexNetSobel, self).__init__()
        self.num_classes = num_classes
        self.alexnet = AlexNet(input_channels=2, num_classes=num_classes, batch_norm_momentum=batch_norm_momentum)
        self.is_sobel = True
        self.sobel = SobelFilter()
        self.features = self.alexnet.features
        self.classifier = self.alexnet.classifier

    def forward(self, x):
        x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetTruncated(nn.Module):
    def __init__(self, net, layer_name):
        super(AlexNetTruncated, self).__init__()
        allowed_layers = ['conv{}'.format(i) for i in xrange(1, 6)] + ['pool5'] \
                        + ['fc{}'.format(i) for i in xrange(6, 8)]
        if layer_name not in allowed_layers:
            raise ValueError('Unknown layer: {}. Only layers from {} allowed'.format(layer_name, allowed_layers))

        self.is_sobel = net.is_sobel
        layers = list()
        if self.is_sobel:
            layers.append(('sobel', net.sobel))
        if layer_name.startswith('fc'):
            feature_layers = net.features._modules.items()
            fc_layer_offset = net.classifier._modules.keys().index(layer_name)
            fc_layers = net.classifier._modules.items()[:fc_layer_offset + 1]
            layers += feature_layers + [('ravel', RavelTensor())] + fc_layers
        else:
            layers += net.features._modules.items()[:AlexNet.feature_layer_index[layer_name] + 1]
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)


def get_output_tensor_shape(net):
    img = np.asarray(Image.fromarray(scipy.misc.face()).resize((224, 224)))
    img = ToTensor()(img).cuda()
    imgs = torch.stack((img, img))
    if torch.cuda.is_available():
        imgs = imgs.cuda()
    return net.forward(imgs).shape[1:]


class AlexNetLinear(nn.Module):
    def __init__(self, net, layer_name, batch_norm_momentum=0.01):
        """

        Args:
            net: full network (Alexnet or
            layer_name: which layer to append the linear classifier to
        """
        super(AlexNetLinear, self).__init__()
        self.num_classes = net.num_classes
        self.is_sobel = net.is_sobel
        self.base_net = AlexNetTruncated(net, layer_name=layer_name)
        num_features = np.prod(get_output_tensor_shape(self.base_net))

        self.linear = nn.Sequential(OrderedDict([
            ('ravel', RavelTensor()),
            ('bn', nn.BatchNorm1d(num_features, momentum=batch_norm_momentum)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc', nn.Linear(num_features, self.num_classes))
        ]))

    def forward(self, x):
        with torch.no_grad():
            x = self.base_net(x)
        x = self.linear(x)
        return x


def get_plain_modules_list(root_module):
    """

    Args:
        modules (OrderedDict):

    Returns:
        module list (without Sequential modules).
    """
    keys = set()
    plain_modules = OrderedDict()

    def add_module(name_, module_):
        if name_ not in keys:
            plain_modules[name_] = module_
            keys.add(name_)
        else:
            raise ValueError('Duplicate module name:', name_)

    for name, module in root_module._modules.items():
        if module.__class__ == nn.Sequential:
            cur_plain_modules = get_plain_modules_list(module)
            for x, y in cur_plain_modules.items():
                add_module(x, y)
        else:
            add_module(name, module)
    return plain_modules


if __name__ == '__main__':
    # net = AlexNet()
    net2 = AlexNetSobel()

    # print AlexNetSobelFc7(net2)
    conv4Net = AlexNetLinear(net2)
