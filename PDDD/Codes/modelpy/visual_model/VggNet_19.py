import torch
import torch.nn as nn


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.model = nn.Sequential(
            #1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),
            #2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),
            #4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2,stride=2),

            #5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),
            #6
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #7
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #8
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(stride=2, kernel_size=2),

            #9
            nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #10
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #11
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #12
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(stride=2, kernel_size=2),

            #13
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #14
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #15
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            #16
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # linear
            nn.Flatten(),
            #17
            nn.Linear(in_features=25088, out_features=2048),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5,),
            #18
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5,),
            #19
            nn.Linear(in_features=512, out_features=117),
            nn.ReLU(True),
            nn.Linear(in_features=117, out_features=1024),
        )

    def forward(self, input):
        x = self.model(input)

        return x
