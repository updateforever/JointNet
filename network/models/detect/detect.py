from network.utils import IntermediateLayerGetter
from network.backbone import (
    resnet,
)
from .centernet import Decoder, Head, CenterNet


def det_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
    else:
        replace_stride_with_dilation = [False, False, False]  # 检测先不减下采样

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    if name == 'centernet':
        return_layers = {'layer4': 'layer4', 'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3'}
        decoder = Decoder(2048)
        head = Head(channel=64, num_classes=num_classes)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = CenterNet(backbone, decoder, head)
    return model
