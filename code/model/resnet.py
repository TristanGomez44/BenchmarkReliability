import torch.nn as nn
import torch.utils.model_zoo as model_zoo




'''

Just a modification of the torchvision resnet model to get the before-to-last activation


'''

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1,dilation=1,groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False,dilation=dilation,groups=groups,padding_mode="circular")


def conv1x1(in_planes, out_planes, stride=1,groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,dilation=1,groups=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,dilation,groups=groups)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes,groups=groups)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inp):

        x = inp

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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,dilation=1):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride,dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identityDown = self.downsample(identity)
            out += identityDown
        else:
            out += identity
        
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, norm_layer=None,stride=2,\
                    strideLay2=2,strideLay3=2,strideLay4=2,chan=64,inChan=3,dilation=1):

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if not type(chan) is list:
            chan = [chan,chan*2,chan*4,chan*8]

        self.inplanes = chan[0]

        self.conv1 = nn.Conv2d(inChan, chan[0], kernel_size=7, stride=stride,bias=False,padding=3)
        self.bn1 = norm_layer(chan[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        self.downsample_ratio = (stride**2)*strideLay2*strideLay3*strideLay4

        if type(dilation) is int:
            dilation = [dilation,dilation,dilation]
        elif len(dilation) != 3:
            raise ValueError("dilation must be a list of 3 int or an int.")

        #All layers are built but they will not necessarily be used
        self.layer1 = self._make_layer(block, chan[0], layers[0], stride=1,norm_layer=norm_layer,dilation=1)
        self.layer2 = self._make_layer(block, chan[1], layers[1], stride=strideLay2, norm_layer=norm_layer,dilation=dilation[0])
        self.layer3 = self._make_layer(block, chan[2], layers[2], stride=strideLay3, norm_layer=norm_layer,dilation=dilation[1])
        self.layer4 = self._make_layer(block, chan[3], layers[3], stride=strideLay4, norm_layer=norm_layer,dilation=dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None,dilation=1):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self,xInp):

        x = self.conv1(xInp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {"feat":x4}

def removeTopLayer(params):
    params.pop("fc.weight")
    params.pop("fc.bias")
    return params

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)

        paramsToLoad = {}
        for key in params:
            if key in model.state_dict() and model.state_dict()[key].size() == params[key].size():
                paramsToLoad[key] = params[key]
        params = paramsToLoad

        model.load_state_dict(params,strict=False)
    return model

def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet34'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet50(pretrained=True,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet50'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=True)
    return model

def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet101'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet152'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model
