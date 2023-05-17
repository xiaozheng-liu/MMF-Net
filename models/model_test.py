import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from cnn.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}


class Exactor(nn.Module):
    def __init__(self):
        super().__init__()
        cnn = psp_models['resnet34'.lower()]()
        self.cnn_pre_stages = nn.Sequential(
            cnn.feats.conv1,  # stride = 2, [bs, c, 240, 320]
            cnn.feats.bn1, cnn.feats.relu,
            cnn.feats.maxpool  # stride = 2, [bs, 64, 120, 160]
        )
        # ####################### downsample stages#######################
        self.cnn_ds_stages = nn.ModuleList([
            cnn.feats.layer1,  # stride = 1, [bs, 64, 120, 160]
            cnn.feats.layer2,  # stride = 2, [bs, 128, 60, 80]
            # stride = 1, [bs, 128, 60, 80]
            nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
            nn.Sequential(cnn.psp, cnn.drop_1)  # [bs, 1024, 60, 80]
        ])
        # ###################### upsample stages #############################
        self.cnn_up_stages = nn.ModuleList([
            nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
            nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
            # nn.Sequential(cnn.final),  # [bs, 64, 240, 320]
        ])

    def forward(self, color):

        # ###################### prepare stages ####################
        # ResNet pre + layer1 + layer2
        rgb_emb = self.cnn_pre_stages(color)  # input (2,3,360,640)   # (2, 64, 90, 160)  缩小1/4

        # ###################### encoding stages #############################
        for i_ds in range(4):
            rgb_emb = self.cnn_ds_stages[i_ds](rgb_emb)

        # ###################### decoding stages #############################
        for i_up in range(2):
            rgb_emb = self.cnn_up_stages[i_up](rgb_emb)
        rgb_emb = F.interpolate(rgb_emb, size=(720, 1280), mode='bilinear', align_corners=True)
        return rgb_emb  # (1,128,20000) / (1,20000,3) / dict


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Conv1d(640, 320, 1),
            nn.ReLU(),
            nn.Conv1d(320, 640, 1)
        )

        self.sigmod = nn.Sigmoid()

        self.mlp2 = nn.Sequential(
            nn.Conv1d(640, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

    def forward(self, color_features, point_features):
        fuse_ori = torch.cat((color_features, point_features), dim=1)
        fuse = self.mlp1(fuse_ori)
        fuse = fuse.transpose(2, 1)
        fuse = nn.AdaptiveMaxPool1d(1)(fuse).squeeze(2)
        att = self.sigmod(fuse)
        fused_feature = fuse_ori * att
        fused_feature = self.mlp2(fused_feature)

        return fused_feature



class IA_Layer(nn.Module):
    def __init__(self, channels):
        super(IA_Layer, self).__init__()
        self.ic, self.pc = [128, 512]
        rc = 320
        self.fc1 = nn.Linear(self.ic, 64)
        self.fc2 = nn.Linear(self.pc, 256)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feas, point_feas):  # (1,64,4096) (1,96,4096)
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # input:(1,64,4096) output:(4096,64) BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # input:(1,96,4096) output:(4096,96)BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)  # input:(4096,64)  output:(4096,24)
        rp = self.fc2(point_feas_f)  # input:(4096,96)  output:(4096,24)
        fused = torch.cat((ri, rp), dim=1)
        att = F.sigmoid(self.fc3(F.tanh(fused)))  # # input:(4096,24) output:(4096,1) BNx1
        att = att.squeeze(1)  # (4091,)
        att = att.view(batch, 1, -1)  # B1N (1,1,4096)

        return att


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels=[inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_I, outplanes, 1)
        # self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)
        att = self.IA_Layer(img_features, point_features)  # (B, 1, num)

        fusion_features = torch.cat([point_features, img_features], dim=1)  # input (1,96,4096) (1,96,4096) output:(1,192,4096)
        fusion_features = fusion_features*att

        fusion_features = self.conv1(fusion_features)  # input:(1,192,4096) output:(1,96,4096)

        return fusion_features

if __name__ == '__main__':
    model = Atten_Fusion_Conv(inplanes_I=128, inplanes_P=512, outplanes=512)
    model = model.cuda()
    img = torch.randn(2, 128, 5000).cuda()
    point = torch.randn(2, 512, 5000).cuda()
    pred = model(point, img)  # out (2, 128, 720, 1280)
    print(pred)

    # model = ChannelAttentionModule(128)
    # model = model.cuda()
    # x = torch.randn(2, 128, 5000).cuda()
    # pred = model(x)  # out (2, 128, 720, 1280)
    # print(pred)
    # x = torch.randn(4096, 24)
    # y = torch.randn(4096, 64)
    # c = x+y
    # print(c)