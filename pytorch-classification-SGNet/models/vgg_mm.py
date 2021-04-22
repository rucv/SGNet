import torch
import torch.nn as nn

cfg_base = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M'],#, 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

cfg_major = [512, 512, 'M']

cfg_minor =  [512, 512, 512, 'M']

class VGG_MM(nn.Module):

    def __init__(self, features, num_class=100, num_major=20):
        super().__init__()
        self.features = features[0]
        self.major_branch = features[1]
        self.minor_branch = features[2]

        self.classifier_major = nn.Sequential(
            nn.Conv2d(512, num_major, kernel_size=1)
        )

        self.classifier_minor = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.features(x)

        x_major = self.major_branch(x)
        pred_major = self.classifier_major(x_major)
        pred_major = pred_major.view(pred_major.size()[0], -1)

        #import pdb; pdb.set_trace()

        x = self.minor_branch(x)
        x = x.view(x.size()[0], -1)
        x_major = x_major.view(x_major.size()[0], -1)
        x = torch.cat((x, x_major), 1)
        pred_minor = self.classifier_minor(x)


        #import pdb; pdb.set_trace()
        #output = output.view(output.size()[0], -1)
        #output = self.classifier(output)
        #import pdb; pdb.set_trace()

        return pred_major, pred_minor

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    bch_major = nn.Sequential(
            nn.Conv2d(512, cfg_major[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg_major[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg_major[0], cfg_major[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg_major[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
    )


    bch_minor = nn.Sequential(
            nn.Conv2d(512, cfg_minor[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg_minor[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg_minor[0], cfg_minor[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg_minor[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg_minor[1], cfg_minor[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg_minor[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
    )

    return [nn.Sequential(*layers), bch_major, bch_minor]

def vgg11_bn():
    return VGG_MM(make_layers(cfg_base['A'], batch_norm=True))

def vgg13_bn():
    return VGG_MM(make_layers(cfg_base['B'], batch_norm=True))

def vgg16_bn(num_cls=100, num_mj=20):
    return VGG_MM(make_layers(cfg_base['D'], batch_norm=True), num_class=num_cls, num_major=num_mj)

def vgg19_bn():
    return VGG_MM(make_layers(cfg_base['E'], batch_norm=True))
