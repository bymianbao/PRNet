import torch
import torch.nn as nn
from Models._module.MDA import _MDA
from Models._module.PLA import _PLA

class PhModel(nn.Module):

    def __init__(self, num_classes=2, pretrained=False):
        super(PhModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.PLA1 = _PLA(inchannels=16, intemp=16)
        self.MDA1 = _MDA(in_dim=8)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2))
        self.PLA2 = _PLA(inchannels=32, intemp=8)
        self.MDA2 = _MDA(in_dim=16)

        self.conv3a = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc6 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x1, cs1, fs1, ss1 = self.PLA1(x)
        x2 = self.MDA1(x, cs1, fs1, ss1)
        x = x1 + x2

        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x1, cs2, fs2, ss2 = self.PLA2(x)
        x2 = self.MDA2(x, cs2, fs2, ss2)
        x = x1 + x2

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = self.global_pool(x)
        x = x.view(-1, 128)
        x = self.fc6(x)
        return x

if __name__ == "__main__":
    inputs = torch.rand(2, 1, 16, 112, 112)
    net = PhModel(num_classes=2, pretrained=False)
    outputs = net.forward(inputs)
    print(outputs.size())
