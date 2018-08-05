import torch
import torch.nn.functional as F
from torch import nn

from resnext import ResNeXt101


class _AttentionModule(nn.Module):
    def __init__(self):
        super(_AttentionModule, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=2, padding=2, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=3, padding=3, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=4, padding=4, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.down = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )

    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = F.relu(self.block2(block1) + block1, True)
        block3 = F.sigmoid(self.block3(block2) + self.down(block2))
        return block3


class BDRAR(nn.Module):
    def __init__(self):
        super(BDRAR, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.refine3_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine2_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine1_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.attention3_hl = _AttentionModule()
        self.attention2_hl = _AttentionModule()
        self.attention1_hl = _AttentionModule()

        self.refine2_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine4_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine3_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.attention2_lh = _AttentionModule()
        self.attention3_lh = _AttentionModule()
        self.attention4_lh = _AttentionModule()

        self.fuse_attention = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 2, 1)
        )

        self.predict = nn.Sequential(
            nn.Conv2d(32, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)

        down4 = F.upsample(down4, size=down3.size()[2:], mode='bilinear')
        refine3_hl_0 = F.relu(self.refine3_hl(torch.cat((down4, down3), 1)) + down4, True)
        refine3_hl_0 = (1 + self.attention3_hl(torch.cat((down4, down3), 1))) * refine3_hl_0
        refine3_hl_1 = F.relu(self.refine3_hl(torch.cat((refine3_hl_0, down3), 1)) + refine3_hl_0, True)
        refine3_hl_1 = (1 + self.attention3_hl(torch.cat((refine3_hl_0, down3), 1))) * refine3_hl_1

        refine3_hl_1 = F.upsample(refine3_hl_1, size=down2.size()[2:], mode='bilinear')
        refine2_hl_0 = F.relu(self.refine2_hl(torch.cat((refine3_hl_1, down2), 1)) + refine3_hl_1, True)
        refine2_hl_0 = (1 + self.attention2_hl(torch.cat((refine3_hl_1, down2), 1))) * refine2_hl_0
        refine2_hl_1 = F.relu(self.refine2_hl(torch.cat((refine2_hl_0, down2), 1)) + refine2_hl_0, True)
        refine2_hl_1 = (1 + self.attention2_hl(torch.cat((refine2_hl_0, down2), 1))) * refine2_hl_1

        refine2_hl_1 = F.upsample(refine2_hl_1, size=down1.size()[2:], mode='bilinear')
        refine1_hl_0 = F.relu(self.refine1_hl(torch.cat((refine2_hl_1, down1), 1)) + refine2_hl_1, True)
        refine1_hl_0 = (1 + self.attention1_hl(torch.cat((refine2_hl_1, down1), 1))) * refine1_hl_0
        refine1_hl_1 = F.relu(self.refine1_hl(torch.cat((refine1_hl_0, down1), 1)) + refine1_hl_0, True)
        refine1_hl_1 = (1 + self.attention1_hl(torch.cat((refine1_hl_0, down1), 1))) * refine1_hl_1

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        refine2_lh_0 = F.relu(self.refine2_lh(torch.cat((down1, down2), 1)) + down1, True)
        refine2_lh_0 = (1 + self.attention2_lh(torch.cat((down1, down2), 1))) * refine2_lh_0
        refine2_lh_1 = F.relu(self.refine2_lh(torch.cat((refine2_lh_0, down2), 1)) + refine2_lh_0, True)
        refine2_lh_1 = (1 + self.attention2_lh(torch.cat((refine2_lh_0, down2), 1))) * refine2_lh_1

        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        refine3_lh_0 = F.relu(self.refine3_lh(torch.cat((refine2_lh_1, down3), 1)) + refine2_lh_1, True)
        refine3_lh_0 = (1 + self.attention3_lh(torch.cat((refine2_lh_1, down3), 1))) * refine3_lh_0
        refine3_lh_1 = F.relu(self.refine3_lh(torch.cat((refine3_lh_0, down3), 1)) + refine3_lh_0, True)
        refine3_lh_1 = (1 + self.attention3_lh(torch.cat((refine3_lh_0, down3), 1))) * refine3_lh_1

        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')
        refine4_lh_0 = F.relu(self.refine4_lh(torch.cat((refine3_lh_1, down4), 1)) + refine3_lh_1, True)
        refine4_lh_0 = (1 + self.attention4_lh(torch.cat((refine3_lh_1, down4), 1))) * refine4_lh_0
        refine4_lh_1 = F.relu(self.refine4_lh(torch.cat((refine4_lh_0, down4), 1)) + refine4_lh_0, True)
        refine4_lh_1 = (1 + self.attention4_lh(torch.cat((refine4_lh_0, down4), 1))) * refine4_lh_1

        refine3_hl_1 = F.upsample(refine3_hl_1, size=down1.size()[2:], mode='bilinear')
        predict4_hl = self.predict(down4)
        predict3_hl = self.predict(refine3_hl_1)
        predict2_hl = self.predict(refine2_hl_1)
        predict1_hl = self.predict(refine1_hl_1)

        predict1_lh = self.predict(down1)
        predict2_lh = self.predict(refine2_lh_1)
        predict3_lh = self.predict(refine3_lh_1)
        predict4_lh = self.predict(refine4_lh_1)

        fuse_attention = F.sigmoid(self.fuse_attention(torch.cat((refine1_hl_1, refine4_lh_1), 1)))
        fuse_predict = torch.sum(fuse_attention * torch.cat((predict1_hl, predict4_lh), 1), 1, True)

        predict4_hl = F.upsample(predict4_hl, size=x.size()[2:], mode='bilinear')
        predict3_hl = F.upsample(predict3_hl, size=x.size()[2:], mode='bilinear')
        predict2_hl = F.upsample(predict2_hl, size=x.size()[2:], mode='bilinear')
        predict1_hl = F.upsample(predict1_hl, size=x.size()[2:], mode='bilinear')
        predict1_lh = F.upsample(predict1_lh, size=x.size()[2:], mode='bilinear')
        predict2_lh = F.upsample(predict2_lh, size=x.size()[2:], mode='bilinear')
        predict3_lh = F.upsample(predict3_lh, size=x.size()[2:], mode='bilinear')
        predict4_lh = F.upsample(predict4_lh, size=x.size()[2:], mode='bilinear')
        fuse_predict = F.upsample(fuse_predict, size=x.size()[2:], mode='bilinear')

        if self.training:
            return fuse_predict, predict1_hl, predict2_hl, predict3_hl, predict4_hl, predict1_lh, predict2_lh, predict3_lh, predict4_lh
        return F.sigmoid(fuse_predict)
