from collections import OrderedDict
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Modified Alexnet with BatchNorm after every conv and FC layer except the last FC layer.

    conv-batchnorm-relu
    fc-batchnorm-relu
    """
    def __init__(self, input_channels=3, num_classes=1000, batch_norm_momentum=0.01):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
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
            ('dropout7', nn.Dropout()),
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


if __name__ == '__main__':
    net = AlexNet()
