import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def unconv1x1(in_plances, out_planes, stride=1):
    """1x1 unconvolution"""
    return nn.ConvTranspose2d(in_plances, out_planes, kernel_size=1, stride=stride, bias=False)


def unconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                              dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class UnpoolBlock(nn.Module):
    def __init__(self, inplanes, planes1, stride=1):
        super(UnpoolBlock, self).__init__()
        self.unpool = unconv3x3(inplanes, planes1, stride=stride)
        self.layer1 = conv1x1(planes1, planes1)
        self.layer2 = conv3x3(planes1, planes1)

    def forward(self, x, output_size=None):
        y1 = self.unpool(x, output_size=output_size)
        y2 = self.layer1(y1)
        y3 = self.layer2(y2)
        return y3


class EAST(nn.Module):
    unpool1_out_planes = 128
    unpool2_out_planes = 64
    unpool3_out_planes = 32
    unpool4_out_planes = 16

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, text_scale=512):
        super(EAST, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.text_scale = text_scale

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.unpool1 = UnpoolBlock(self.inplanes, EAST.unpool1_out_planes, 2)

        self.inplanes = EAST.unpool1_out_planes * 3

        self.unpool2 = UnpoolBlock(self.inplanes, EAST.unpool2_out_planes, 2)

        self.inplanes = EAST.unpool2_out_planes * 3

        self.unpool3 = UnpoolBlock(self.inplanes, EAST.unpool3_out_planes, 2)

        self.inplanes = EAST.unpool3_out_planes

        self.layer5 = conv3x3(self.inplanes, EAST.unpool3_out_planes)

        self.layer6 = conv1x1(self.inplanes, 6)

        self.activation = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    @staticmethod
    def _align_output_size(target_size, out_planes):
        return [target_size[0], out_planes, target_size[2], target_size[3]]

    def _forward_impl(self, x, ground_truth, training_mask):
        # See note [TorchScript super()]
        x_0 = self.conv1(x)
        x_1 = self.bn1(x_0)
        x_2 = self.relu(x_1)
        x0 = self.maxpool(x_2)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        y1 = self.unpool1(x4, output_size=EAST._align_output_size(x3.size(), EAST.unpool1_out_planes * 2))
        y_in2 = torch.cat((y1, x3), dim=1)
        y2 = self.unpool2(y_in2, output_size=EAST._align_output_size(x2.size(), EAST.unpool2_out_planes * 2))
        y_in3 = torch.cat((y2, x2), dim=1)
        y3 = self.unpool3(y_in3, output_size=EAST._align_output_size(x1.size(), EAST.unpool3_out_planes * 2))

        y4 = self.layer5(y3)
        y5 = self.layer6(y4)

        y6 = self.activation(y5)

        score_map = y6[:, 0:1]

        geo_map = y6[:, 1:5] * self.text_scale

        angle_map = (y6[:, 5:] - 0.5) * np.pi / 2

        res = (score_map, geo_map, angle_map)

        loss = None

        if ground_truth is not None:
            loss = self.calculate_loss(res, ground_truth, training_mask=training_mask)

        return res, loss

    def forward(self, x, ground_truth=None, training_mask=None):
        return self._forward_impl(x, ground_truth, training_mask)

    def calculate_loss(self, calcuation, ground_truth, training_mask):
        score_map, geo_map, angle_map = calcuation
        gt_score_map, gt_geo_map, gt_angle_map = ground_truth
        if training_mask is None:
            training_mask = torch.ones(score_map.size())

        eps = 1e-5
        loss_score = 1.0 - 2.0 * (score_map * training_mask * gt_score_map).sum() / (
                    ((score_map + gt_score_map) * training_mask).sum() + eps)
        w_union = torch.min(geo_map[:, 0], gt_geo_map[:, 0]) + torch.min(geo_map[:, 2], gt_geo_map[:, 2])
        h_union = torch.min(geo_map[:, 1], gt_geo_map[:, 1]) + torch.min(geo_map[:, 3], gt_geo_map[:, 3])

        area = (geo_map[:, 0] + geo_map[:, 2]) * (geo_map[:, 1] + geo_map[:, 3])
        gt_area = (gt_geo_map[:, 0] + gt_geo_map[:, 2]) * (gt_geo_map[:, 1] + gt_geo_map[:, 3])
        intersaction = w_union * h_union
        union = area + gt_area - intersaction
        l_aabb = -torch.log((intersaction + 1.0) / (union + 1.0))
        l_angle = 1 - torch.cos(angle_map - gt_angle_map)
        l_g = l_aabb + 20 * l_angle
        return (l_g * gt_score_map * training_mask).sum()/training_mask.sum() + loss_score * 0.01


def _resnet(model_location, block, layers, pretrained, **kwargs):
    model = EAST(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load(model_location)
        model.load_state_dict(state_dict)
    return model


def east_reset18(pretrained_location=None, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(pretrained_location, BasicBlock, [2, 2, 2, 2], pretrained_location is not None,
                   **kwargs)
