from torch import nn
import os
import torch

# 参数配置,标准的darknet19参数.
# Darknet19
cfg = [32, 'M', 64, 'M', 128, 64, 128, 'M', 256, 128, 256, 'M',
       512, 256, 512, 256, 512, 'M', 1024, 512, 1024, 512, 1024]

# 定义残差模块，包含两个卷积层和一个残差连接
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层，使用1x1卷积，降低通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.relu1 = nn.LeakyReLU(0.1)
        # 第二个卷积层，使用3x3卷积，恢复通道数
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        # 保存输入作为残差
        residual = x
        # 通过两个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # 加上残差
        out += residual
        return out

def make_layers(cfg, in_channels=3, batch_norm=True):
    """
    从配置参数中构建网络
    :param cfg:  参数配置
    :param in_channels: 输入通道数,RGB彩图为3, 灰度图为1
    :param batch_norm:  是否使用批正则化
    :return:
    """
    layers = []
    flag = True             # 用于变换卷积核大小,(True选后面的,False选前面的)
    in_channels= in_channels
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels = in_channels,
                                   out_channels= v,
                                   kernel_size=(1, 3)[flag],
                                   stride=1,
                                   padding=(0,1)[flag],
                                   bias=False))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            in_channels = v

            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        flag = not flag

    return nn.Sequential(*layers)

class DarkNet19(nn.Module):
    """
    Darknet19 模型
    """
    def __init__(self, num_classes=1000, in_channels=3, batch_norm=True, init_weights=False):
        """
        模型结构初始化
        :param num_classes: 最终分类数       (nums of classification.)
        :param in_channels: 输入数据的通道数  (input pic`s channel.)
        :param batch_norm:  是否使用正则化    (use batch_norm, True or False;True by default.)
        :param pretrained:  是否导入预训练参数 (use the pretrained weight)
        """
        super(DarkNet19, self).__init__()
        # 调用nake_layers 方法搭建网络
        # (build the network)
        self.features = make_layers(cfg, in_channels=in_channels, batch_norm=batch_norm)
        self.fc = nn.Linear(50176, 1024)
        # 网络最后的分类层,使用 [1x1卷积和全局平均池化] 代替全连接层.
        # (use 1x1 Conv and averagepool replace the full connection layer.)
#        self.classifier = nn.Sequential(
#            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1),
#            nn.AdaptiveAvgPool2d(output_size=(1)),
#            nn.Softmax(dim=0)
#        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
#        x = self.classifier(x)
        x = x.view(x.size(0),-1)
#        print(x.shape)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class DarkNet53(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, init_weights=False):
        super(DarkNet53, self).__init__()
        # 第一个卷积层，使用3x3卷积，输出通道数为32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(0.1)
        # 第一个残差层组，包含一个下采样卷积层和一个残差模块，输出通道数为64
        self.layer1 = self._make_layer(32, 64, 1)
        # 第二个残差层组，包含一个下采样卷积层和两个残差模块，输出通道数为128
        self.layer2 = self._make_layer(64, 128, 2)
        # 第三个残差层组，包含一个下采样卷积层和八个残差模块，输出通道数为256
        self.layer3 = self._make_layer(128, 256, 8)
        # 第四个残差层组，包含一个下采样卷积层和八个残差模块，输出通道数为512
        self.layer4 = self._make_layer(256, 512, 8)
        # 第五个残差层组，包含一个下采样卷积层和四个残差模块，输出通道数为1024
        self.layer5 = self._make_layer(512, 1024, 4)
        self.fc = nn.Linear(50176, 1024)
  #      self.classifier = nn.Sequential(
 #           nn.Conv2d(1024, num_classes, kernel_size=1, stride=1),
 #           nn.AdaptiveAvgPool2d(output_size=(1)),
  #          nn.Softmax(dim=0)
  #      )
        
        if init_weights:
            self._initialize_weights()
        
        # 定义一个辅助函数，用于构造残差层组
    def _make_layer(self, in_channels, out_channels, blocks):
        # 创建一个空的列表，用于存放网络层
        layers = []
        # 首先添加一个下采样卷积层，使用3x3卷积，步长为2，输出通道数为out_channels
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.1))
        # 然后添加blocks个残差模块，输入和输出通道数都为out_channels
        for i in range(blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        # 将列表转换为顺序容器，并返回
        return nn.Sequential(*layers)

        # 定义前向传播函数

    def forward(self, x):
        # 通过第一个卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # 通过第一个残差层组
        x = self.layer1(x)
        # 通过第二个残差层组
        x = self.layer2(x)
        # 通过第三个残差层组，并保存输出作为out3
        x = self.layer3(x)
        # 通过第四个残差层组，并保存输出作为out4
        x = self.layer4(x)
        # 通过第五个残差层组，并保存输出作为out5
        x = self.layer5(x)
       
#        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        # 返回三个不同尺度的输出特征图
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
