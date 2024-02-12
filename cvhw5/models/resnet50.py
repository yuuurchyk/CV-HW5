from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck


def get_resnet50(n_classes: int = 1000, width_multiplier: int = 1) -> ResNet:
    assert width_multiplier >= 1, f'Wrong width multiplier: {width_multiplier}'
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes, width_per_group=64 * width_multiplier)
