from .joint import *


class JointNetV1(nn.Module):
    def __init__(self, backbone, decoder, head, seg_head):
        super(JointNetV1, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.head = head
        self.seg_head = seg_head

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        seg_feats = self.seg_head(features)
        # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        det_feat = self.decoder(features, seg_feats)
        output = self.head(seg_feats, det_feat)
        return output

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


def joint_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, False]  # 检测先不减下采样
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    return_layers = {'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'}
    decoder = JointDecoder(2048)
    head = JointHead(channel=64, num_classes=num_classes)
    seg_head = DeepLabHeadV3Plus(2048, 256, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = JointNetV1(backbone, decoder, head, seg_head)
    return model
