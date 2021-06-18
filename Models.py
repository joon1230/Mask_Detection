from torchvision.models import resnet34
from torchvision.models import resnet50
import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT
import torch
import timm


class Resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = resnet50(pretrained=pretrained)
        #         self.model = timm.create_model('inception_resnet_v2', pretrained = True)
        self.model.fc = nn.Linear(2048, 18)

    #         self.model.classif = nn.Linear(1536, 18)
    def forward(self, x):
        x = self.model(x)
        return x


class ResNext(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnext50_32x4d', pretrained=True)
        self.model.fc = nn.Linear(2048, 18)
    def forward(self, x):
        x = self.model(x)
        return x


class InceptionResnetv2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('inception_resnet_v2', pretrained=pretrained)
        self.model.classif = nn.Linear(1536, 18)

    def forward(self, x):
        x = self.model(x)
        return x


class Resnet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = resnet34(pretrained=pretrained)
        self.model.fc = nn.Linear(512, 18)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNet_b0(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.model.classifier = nn.Linear(1280, 18)

    def forward(self, x):
        y = self.model(x)
        return y


class EfficientNet_b2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        self.model.classifier = nn.Linear(1408, 18)

    def forward(self, x):
        y = self.model(x)
        return y


class EfficientNet_b2_renew(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        self.model.classifier = nn.Sequential( nn.Linear(1408,256), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(256, 18) )

    def forward(self, x):
        y = self.model(x)
        return y


class EfficientNet_b2_renew_ver2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        self.classifier = nn.Sequential( nn.Linear(1000,256), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(256, 18) )

    def forward(self, x):
        y = self.model(x)
        y = self.classifier(y)
        return y


class EfficientNet_b7(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained("efficientnet-b7")
            self.model._fc = nn.Linear(2560, 18)
        else:
            self.model = EfficientNet.from_name("efficientnet-b7")
            self.model._fc = nn.Linear(2560, 18)

    def forward(self, x):
        y = self.model(x)
        return y


class EfficientNet_b4(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained("efficientnet-b4")
            self.model._fc = nn.Linear(1792, 18)
        else:
            self.model = EfficientNet.from_name('efficientnet-b4')
            self.model._fc = nn.Linear(1792, 18)

    def forward(self, x):
        y = self.model(x)
        return y


class EfficientNet_b4_renew(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained("efficientnet-b4")
            self.model._fc = nn.Sequential( nn.Linear(1792,256), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(256, 18) )
        else:
            self.model = EfficientNet.from_name('efficientnet-b4')
            self.model._fc = nn.Sequential( nn.Linear(1792,256), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(256, 18) )

    def forward(self, x):
        y = self.model(x)
        return y


class EfficientNet_b4_timm(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = EfficientNet.from_name("efficientnet-b4")
        self.fc = nn.Sequential( nn.Linear(1000,256), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(256, 18) )
        # self.set_freeze()

    def forward(self, x):
        y = self.backbone(x)
        y = self.fc(y)
        return y


class VisionTransform(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = ViT('B_16_imagenet1k', pretrained=True)
        else:
            self.model = ViT('B_16_imagenet1k', pretrained=False)
        self.fc = nn.Linear(768, 18)
    def forward(self,x):
        y = self.model(x)
        return y


class EfficientNet_b2_raw(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        self.model.classifier = nn.Linear(1408, 18)

    def forward(self, x):
        y = self.model(x)
        return y

