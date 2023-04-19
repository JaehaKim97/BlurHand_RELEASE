import torch
import torch.nn as nn

from models.modules.layer_utils import make_linear_layers
from utils.human_models.human_models import mano
from utils.transforms import rot6d_to_axis_angle

# for applying KTD
ANCESTOR_INDEX = [
    [],  # Wrist
    [0],  # Index_1
    [0,1],  # Index_2
    [0,1,2],  # index_3
    [0],  # Middle_1
    [0,4],  # Middle_2
    [0,4,5],  # Middle_3
    [0],  # Pinky_1
    [0,7],  # Pinky_2
    [0,7,8],  #Pinky_3
    [0],  # Ring_1
    [0,10],  # Ring_2
    [0,10,11],  # Ring_3
    [0],  # Thumb_1
    [0,13],  # Thumb_2
    [0,13,14]  # Thumb_3
]


class Regressor(nn.Module):
    def __init__(self, opt_params, in_chans=2048, in_chans_pose=512):
        super().__init__()
        # mano shape regression, multiply the output channel by 3 to account e1, m, and e2
        self.shape_out = make_linear_layers([in_chans, mano.shape_param_dim * 3], relu_final=False)
        
        # mano pose regression, apply KTD using ancestral pose parameters
        self.joint_regs = nn.ModuleList()
        for ancestor_idx in ANCESTOR_INDEX:
            regressor = nn.Linear(opt_params['num_joints'] * (in_chans_pose+3) + 6 * len(ancestor_idx), 6)
            self.joint_regs.append(regressor)
            
        # camera parameter regression for projection loss, multiply 3 to account e1, m, and e2
        self.cam_out = make_linear_layers([in_chans, 3 * 3], relu_final=False)

    def forward(self, feat_blur, feat_joint_e1, feat_joint_md, feat_joint_e2, joint_img_e1, joint_img_md, joint_img_e2):
        # mano shape parameter regression
        shape_param = self.shape_out(feat_blur.mean((2,3)))
        mano_shape_e1, mano_shape_md, mano_shape_e2 = torch.split(shape_param, mano.shape_param_dim, dim=1)
        
        # mano pose parameter regression
        batch_size = feat_blur.shape[0]
        feat_one_e1 = torch.cat((feat_joint_e1, joint_img_e1), 2).reshape(batch_size, -1)
        feat_one_md = torch.cat((feat_joint_md, joint_img_md), 2).reshape(batch_size, -1)
        feat_one_e2 = torch.cat((feat_joint_e2, joint_img_e2), 2).reshape(batch_size, -1)

        pose_6d_list_e1, pose_6d_list_md, pose_6d_list_e2 = [], [], []

        # regression using KTD
        for ancestor_idx, reg in zip(ANCESTOR_INDEX, self.joint_regs):
            ances_e1 = torch.cat([feat_one_e1] + [pose_6d_list_e1[i] for i in ancestor_idx], dim=1)
            pose_6d_list_e1.append(reg(ances_e1))
            
            ances_md = torch.cat([feat_one_md] + [pose_6d_list_md[i] for i in ancestor_idx], dim=1)
            pose_6d_list_md.append(reg(ances_md))
            
            ances_e2 = torch.cat([feat_one_e2] + [pose_6d_list_e2[i] for i in ancestor_idx], dim=1)
            pose_6d_list_e2.append(reg(ances_e2))
            
        pose_6d_e1 = torch.cat(pose_6d_list_e1, dim=1)
        pose_6d_md = torch.cat(pose_6d_list_md, dim=1)
        pose_6d_e2 = torch.cat(pose_6d_list_e2, dim=1)
        
        # change 6d pose -> axis angles
        mano_pose_e1 = rot6d_to_axis_angle(pose_6d_e1.reshape(-1, 6)).reshape(-1, mano.orig_joint_num * 3)
        mano_pose_md = rot6d_to_axis_angle(pose_6d_md.reshape(-1, 6)).reshape(-1, mano.orig_joint_num * 3)
        mano_pose_e2 = rot6d_to_axis_angle(pose_6d_e2.reshape(-1, 6)).reshape(-1, mano.orig_joint_num * 3)
        
        # camera parameter regression
        cam_param = self.cam_out(feat_blur.mean((2,3)))
        cam_param_e1, cam_param_md, cam_param_e2 = torch.split(cam_param, 3, dim=1)

        return mano_shape_e1, mano_shape_md, mano_shape_e2, mano_pose_e1, mano_pose_md, mano_pose_e2, cam_param_e1, cam_param_md, cam_param_e2
