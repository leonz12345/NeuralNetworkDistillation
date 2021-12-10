import torch
import torch.nn as nn
import torch.nn.functional as F

# Implement Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.identity_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        else:
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        if self.downsample:
            identity = self.identity_proj(x)
        else:
            identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out
    
# Implement ResNet 20
class ResNet20(nn.Module):
    def __init__(self):
        super(ResNet20, self).__init__()        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) # In:32x32X3 Out:32x32X16
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.layer(16,16,3,downsample=False)
        self.layer2 = self.layer(16,32,3,downsample=True)
        self.layer3 = self.layer(32,64,3,downsample=True)
        self.global_avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(1*1*64, 10) # Flatten: 1*1*64
        
    def layer(self, in_channels, out_channels, num_of_blocks, downsample=False):
        blocks = []
        if downsample:
            blocks.append(ResidualBlock(in_channels, out_channels, downsample=True))
            for i in range(num_of_blocks-1):
                blocks.append(ResidualBlock(out_channels, out_channels))
        else:
            for i in range(num_of_blocks):
                blocks.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Implement ResNet 50
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.layer(16,16,8,downsample=False)
        self.layer2 = self.layer(16,32,8,downsample=True)
        self.layer3 = self.layer(32,64,8,downsample=True)
        self.global_avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(1*1*64, 10)
        
    def layer(self, in_channels, out_channels, num_of_blocks, downsample=False):
        blocks = []
        if downsample:
            blocks.append(ResidualBlock(in_channels, out_channels, downsample=True))
            for i in range(num_of_blocks-1):
                blocks.append(ResidualBlock(out_channels, out_channels))
        else:
            for i in range(num_of_blocks):
                blocks.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out