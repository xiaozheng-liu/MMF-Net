import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix
from cnn.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}


class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 2, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)
        end_points['objectness_score'] = graspable_score
        return end_points

class Exactor(nn.Module):
    def __init__(self):
        super().__init__()
        cnn = psp_models['resnet18'.lower()]()
        self.cnn_pre_stages = nn.Sequential(
            cnn.feats.conv1,
            cnn.feats.bn1, cnn.feats.relu,
            cnn.feats.maxpool
        )
        # ########## downsample stages#########
        self.cnn_ds_stages = nn.ModuleList([
            cnn.feats.layer1,
            cnn.feats.layer2,
            nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
            nn.Sequential(cnn.psp, cnn.drop_1)
        ])
        # ########## upsample stages ##############
        self.cnn_up_stages = nn.ModuleList([
            nn.Sequential(cnn.up_1, cnn.drop_2),
            nn.Sequential(cnn.up_2, cnn.drop_2),
        ])

    def forward(self, color):
        # ###################### prepare stages ####################
        # ResNet pre + layer1 + layer2
        rgb_emb = self.cnn_pre_stages(color)
        # ###################### encoding stages #############################
        for i_ds in range(4):
            rgb_emb = self.cnn_ds_stages[i_ds](rgb_emb)
        # ###################### decoding stages #############################
        for i_up in range(2):
            rgb_emb = self.cnn_up_stages[i_up](rgb_emb)
        rgb_emb = F.interpolate(rgb_emb, size=(720, 1280), mode='bilinear', align_corners=True)

        return rgb_emb


class Atten_Layer(nn.Module):
    def __init__(self, channels):
        super(Atten_Layer, self).__init__()
        self.ic, self.pc = [128, 512]
        rc = 320
        self.fc1 = nn.Linear(self.ic, 64)
        self.fc2 = nn.Linear(self.pc, 256)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)
        point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        fused = torch.cat((ri, rp), dim=1)
        att = F.sigmoid(self.fc3(F.tanh(fused)))
        att = att.squeeze(1)
        att = att.view(batch, 1, -1)

        return att


class CoGuided_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(CoGuided_Fusion_Conv, self).__init__()

        self.IA_Layer = Atten_Layer(channels=[inplanes_I, inplanes_P])
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_I, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        att = self.IA_Layer(img_features, point_features)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = fusion_features*att
        fusion_features = self.bn1(self.conv1(fusion_features))

        return fusion_features, att


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous()
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            top_view_inds = []
            for i in range(B):
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)
        else:
            _, top_view_inds = torch.max(view_score, dim=2)

            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)
        new_features = self.mlps(grouped_feature)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        new_features = new_features.squeeze(-1)
        return new_features


class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points
