'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PCENet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, num_classes=10):
        super(PCENet, self).__init__()
        self.in_planes = 32

        self.rnn_module = 'GRU'
        self.rnn_layers= 3
        self.bidirectional = True
        self.input_dim = 1024
        self.reduction = 16 #pce_light 32
        self.hidden_dim = self.input_dim // self.reduction

        self.conv1 = nn.Conv1d(in_channels, out_channels=32, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self._make_lateral_layer(dropout_rate=0.0,
        )

        self.fc = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_lateral_layer(self, input_dim=512, dropout_rate=0.0):
        # self.l0_avgpool = nn.AdaptiveAvgPool1d((1, 1))
        self.l1_avgpool = nn.AdaptiveAvgPool1d(1)
        self.l2_avgpool = nn.AdaptiveAvgPool1d(1)
        self.l3_avgpool = nn.AdaptiveAvgPool1d(1)
        self.l4_avgpool = nn.AdaptiveAvgPool1d(1)

        # self.l0_fc = nn.Sequential(
        #     nn.Linear(64, self.input_dim),
        #     nn.ReLU(inplace=True),
        # )
        self.l1_fc = nn.Sequential(
            nn.Linear(128, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.l2_fc = nn.Sequential(
            nn.Linear(256, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.l3_fc = nn.Sequential(
            nn.Linear(512, self.input_dim),
            nn.ReLU(inplace=True),
        )
        # self.l4_fc = nn.Sequential(
        #     nn.Linear(2048, self.input_dim),
        #     nn.ReLU(inplace=True),
        # )

        self.cse =\
            nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.rnn_layers,
                batch_first=True, bidirectional=self.bidirectional,
                dropout=dropout_rate,
            ) if self.rnn_module == 'GRU' else\
            nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.rnn_layers,
                batch_first=True, bidirectional=self.bidirectional,
                dropout=dropout_rate,
            )
        self.cse_fc = nn.Sequential(
            nn.Linear(2 * self.hidden_dim if self.bidirectional else self.hidden_dim, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)

        # out = F.relu(self.bn1(self.conv1(x)))
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x0 = self.maxpool(x) # [bs, C=64, H=56, W=56]

        x1 = self.layer1(x0) # [bs, C=256, H=28, W=28]
        x2 = self.layer2(x1) # [bs, C=512, H=14, W=14]
        x3 = self.layer3(x2) # [bs, C=1024, H=7, W=7]
        x4 = self.layer4(x3) # [bs, C=2048, H=4, W=4]

        # ========= Contextual Squeeze and Excitation (CSE) =========
        # f0 = self.l0_avgpool(x0).view(batch_size, -1) # [bs, C=256]
        f1 = self.l1_avgpool(x1).view(batch_size, -1) # [bs, C=256]
        f2 = self.l2_avgpool(x2).view(batch_size, -1) # [bs, C=512]
        f3 = self.l3_avgpool(x3).view(batch_size, -1) # [bs, C=1024]
        # f4 = self.l4_avgpool(x4).view(batch_size, -1) # [bs, C=2048]

        # f0 = self.l0_fc(f0) # [bs, C=2048]
        f1 = self.l1_fc(f1) # [bs, C=2048]
        f2 = self.l2_fc(f2) # [bs, C=2048]
        f3 = self.l3_fc(f3) # [bs, C=2048]
        # f4 = self.l4_fc(f4) # [bs, C=2048]

        hidden_tuple = (2 * self.rnn_layers if self.bidirectional else self.rnn_layers, batch_size, self.hidden_dim)
        h0 = torch.zeros(hidden_tuple).to(device)
        if self.rnn_module == 'LSTM':
            c0 = torch.zeros(hidden_tuple).to(device)
        self.cse.flatten_parameters()

        # x_cse = torch.stack((f1, f2, f3, f4), dim=1) # [bs, S=4, C=2048]
        x_cse = torch.stack((f1, f2, f3), dim=1) # [bs, S, C=2048]
        if self.rnn_module == 'GRU':
            cse_out, cse_hn = self.cse(x_cse, h0) # [bs, S, C=hidden_dim]
        elif self.rnn_module == 'LSTM':
            cse_out, cse_hn = self.cse(x_cse, (h0, c0)) # [bs, S, C=hidden_dim]

        cse_out = cse_out[:, -1, :] # [bs, C=hidden_dim]
        g_cse = self.cse_fc(cse_out).view(batch_size, -1, 1) # [bs, C=2048, H=1, W=1]

        # Excitation
        x = x4 * g_cse.expand_as(x4) # [bs, C=2048, H=4, W=4]

        x_feat = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x, x_feat


def PCENetLight(in_channels, num_classes):
    return PCENet(Bottleneck, [2, 2, 2, 2], in_channels, num_classes)


def PCENet34(num_classes):
    return PCENet(Bottleneck, [2, 2, 2, 2], num_classes)


def PCENet50(num_classes):
    return PCENet(Bottleneck, [3, 4, 6, 3], num_classes)


def PCENet101(num_classes):
    return PCENet(Bottleneck, [3, 4, 23, 3], num_classes)


def PCENet152(num_classes):
    return PCENet(Bottleneck, [3, 8, 36, 3], num_classes)