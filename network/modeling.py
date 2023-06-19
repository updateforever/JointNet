from .models.joint.joint import joint_resnet
from .models.segment import *
from .models.detect import *


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone == 'mobilenetv2':
        model = segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride,
                               pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                            pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone == 'xception':
        model = segm_xception(arch_type, backbone, num_classes, output_stride=output_stride,
                              pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


def _load_model_det(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone.startswith('resnet'):
        model = det_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                           pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


def _load_model_joint(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone.startswith('resnet'):
        model = joint_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                           pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


# Deeplab v3
def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False):  # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', output_stride, num_classes, pretrained_backbone=pretrained_backbone)


def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3', 'hrnetv2_32', output_stride, num_classes, pretrained_backbone=pretrained_backbone)


def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'xception', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False):  # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def centernet_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a centernet model with a resnet50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model_det('centernet', 'resnet50', num_classes, output_stride=output_stride,
                           pretrained_backbone=pretrained_backbone)


def jointnet_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a centernet model with a resnet50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model_det('jointnet', 'resnet50', num_classes, output_stride=output_stride,
                           pretrained_backbone=pretrained_backbone)
