import torch
from torch import nn

cfgs = {
    '11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_features(in_channels, size='11'):
    tool = []
    channels = in_channels
    for case in cfgs[size]:
        if case == 64:
            tool.append(nn.Conv2d(channels, 64, 3, 1, 1))
            tool.append(nn.ReLU(inplace=True))
            channels = 64
        elif case == 128:
            tool.append(nn.Conv2d(channels, 128, 3, 1, 1))
            tool.append(nn.ReLU(inplace=True))
            channels = 128
        elif case == 256:
            tool.append(nn.Conv2d(channels, 256, 3, 1, 1))
            tool.append(nn.ReLU(inplace=True))
            channels = 256
        elif case == 512:
            tool.append(nn.Conv2d(channels, 512, 3, 1, 1))
            tool.append(nn.ReLU(inplace=True))
            channels = 512
        elif case == 'M':
            tool.append(nn.MaxPool2d(2, 2))
            
    return nn.Sequential(*tool)

class VGG(nn.Module):
    def __init__(self, in_channels, out_channels, size='11') -> None:
        super().__init__()
        self.features = make_features(in_channels, size)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, out_channels)
        )
    
    def forward(self, x):
        feature = self.features(x)
        feature = torch.flatten(feature, start_dim=1)
        return self.classifier(feature)
    
    

