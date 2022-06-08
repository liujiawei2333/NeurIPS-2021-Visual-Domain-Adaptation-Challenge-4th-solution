from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False):
        super(ResBase, self).__init__()
        self.dim = 2048
        self.top = top
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnext101':
            model_ft = models.resnext101_32x8d(pretrained=pret)

        if top:
            self.features = model_ft
        else:
            mod = list(model_ft.children())
            mod.pop()
            self.features = nn.Sequential(*mod)


    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.view(x.size(0), self.dim)
            return x

class EfficientPlus2CNN(nn.Module):
    def __init__(self,option='efficient'):
        super(EfficientPlus2CNN, self).__init__()
        self.efficient = EfficientNet.from_pretrained('efficientnet-b7')
        self.conv1 = nn.Conv2d(in_channels=2560, out_channels=2048, kernel_size=3, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.efficient.extract_features(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        return out

class VGGBase(nn.Module):
    def __init__(self, option='vgg', pret=True, no_pool=False, top=False):
        super(VGGBase, self).__init__()
        self.dim = 2048
        self.no_pool = no_pool
        self.top = top

        if option =='vgg11_bn':
            vgg16=models.vgg11_bn(pretrained=pret)
        elif option == 'vgg11':
            vgg16 = models.vgg11(pretrained=pret)
        elif option == 'vgg13':
            vgg16 = models.vgg13(pretrained=pret)
        elif option == 'vgg13_bn':
            vgg16 = models.vgg13_bn(pretrained=pret)
        elif option == "vgg16":
            vgg16 = models.vgg16(pretrained=pret)
        elif option == "vgg16_bn":
            vgg16 = models.vgg16_bn(pretrained=pret)
        elif option == "vgg19":
            vgg16 = models.vgg19(pretrained=pret)
        elif option == "vgg19_bn":
            vgg16 = models.vgg19_bn(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        if self.top:
            self.vgg = vgg16

    def forward(self, x, source=True,target=False):
        if self.top:
            x = self.vgg(x)
            return x
        else:
            x = self.features(x)
            x = x.view(x.size(0), 7 * 7 * 512)
            x = self.classifier(x)
            return x

class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp
        self.dropout = nn.Dropout(p=0.3)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
            # x = self.dropout(x)
        else:
            x = self.fc(x)
            # x = self.dropout(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)

class ResClassifier_MME_C2(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME_C2, self).__init__()
        if norm:
            self.fc1 = nn.Linear(input_size, num_classes, bias=False)
            self.fc2 = nn.Linear(num_classes, num_classes, bias=False)
            self.fc3 = nn.Linear(num_classes, num_classes, bias=False)
        else:
            self.fc1 = nn.Linear(input_size, num_classes, bias=False)
            self.fc2 = nn.Linear(num_classes, num_classes, bias=False)
            self.fc3 = nn.Linear(num_classes, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)/self.tmp
        else:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)

        return x

    def weight_norm(self):
        w1 = self.fc1.weight.data
        norm1 = w1.norm(p=2, dim=1, keepdim=True)
        w2 = self.fc2.weight.data
        norm2 = w2.norm(p=2, dim=1, keepdim=True)
        w3 = self.fc3.weight.data
        norm3 = w3.norm(p=2, dim=1, keepdim=True)
        self.fc1.weight.data = w1.div(norm1.expand_as(w1))
        self.fc2.weight.data = w2.div(norm2.expand_as(w2))
        self.fc3.weight.data = w3.div(norm3.expand_as(w3))

    def weights_init(self):
        self.fc1.weight.data.normal_(0.0, 0.1)
        self.fc2.weight.data.normal_(0.0, 0.1)
        self.fc3.weight.data.normal_(0.0, 0.1)

