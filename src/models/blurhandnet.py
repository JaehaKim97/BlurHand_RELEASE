import copy
import math
import torch
import torch.nn as nn

from losses import CoordLoss, ParamLoss, CoordLossOrderInvariant
from models.modules.ktformer import KTFormer
from models.modules.regressor import Regressor
from models.modules.resnetbackbone import ResNetBackbone
from models.modules.unfolder import Unfolder
from models.modules.layer_utils import init_weights
from utils.MANO import mano


class BlurHandNet(nn.Module):
    def __init__(self, opt, weight_init=True):
        super().__init__()
        # define trainable module
        opt_net = opt['network']
        self.backbone = ResNetBackbone(**opt_net['backbone'])  # backbone
        self.unfolder = Unfolder(opt['task_parameters'], **opt_net['unfolder'])  #  Unfolder
        self.ktformer = KTFormer(opt['task_parameters'], **opt_net['ktformer'])  # KTFormer
        self.regressor = Regressor(opt['task_parameters'], **opt_net['regressor'])  # Regressor
        self.trainable_modules = [self.backbone, self.unfolder, self.ktformer, self.regressor]
        
        # weight initialization
        if weight_init:
            self.backbone.init_weights()
            self.unfolder.apply(init_weights)
            self.ktformer.apply(init_weights)
            self.regressor.apply(init_weights)
        
        # for producing 3d hand meshs
        self.mano_layer_right = copy.deepcopy(mano.layer['right']).cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left']).cuda()
        
        # losses
        self.coord_loss = CoordLoss()
        self.coord_loss_order_invariant = CoordLossOrderInvariant()
        self.param_loss = ParamLoss()
        
        # parameters
        self.opt_params = opt['task_parameters']
        if opt.get('train', False):
            self.opt_loss = opt['train']['loss']
        
    def forward(self, inputs, targets, meta_info, mode):
        # extract feature from backbone
        feat_blur, feat_pyramid = self.backbone(inputs['img'])
        
        # extract temporal information via Unfolder
        feat_joint_e1, feat_joint_md, feat_joint_e2, joint_img_e1, joint_img_md, joint_img_e2 = \
            self.unfolder(feat_blur, feat_pyramid)
        
        # feature enhancing via KTFormer
        feat_joint_e1, feat_joint_md, feat_joint_e2 = \
            self.ktformer(feat_joint_e1, feat_joint_md, feat_joint_e2)
        
        # regress mano shape, pose and camera parameter
        mano_shape_e1, mano_shape_md, mano_shape_e2, mano_pose_e1, mano_pose_md, mano_pose_e2, cam_param_e1, cam_param_md, cam_param_e2 = \
            self.regressor(feat_blur, feat_joint_e1, feat_joint_md, feat_joint_e2, joint_img_e1.detach(), joint_img_md.detach(), joint_img_e2.detach())
        
        # obtain camera translation to project 3D coordinates into 2D space
        cam_trans_e1, cam_trans_md, cam_trans_e2 = \
            self.get_camera_trans(cam_param_e1), self.get_camera_trans(cam_param_md), self.get_camera_trans(cam_param_e2)
        
        # obtain 1) projected 3D coordinates 2) camera-centered 3D joint coordinates 3) camera-centered 3D meshes
        joint_proj_e1, joint_cam_e1, mesh_cam_e1 = \
            self.get_coord(mano_pose_e1[:,:3], mano_pose_e1[:,3:], mano_shape_e1, cam_trans_e1)
        joint_proj_md, joint_cam_md, mesh_cam_md = \
            self.get_coord(mano_pose_md[:,:3], mano_pose_md[:,3:], mano_shape_md, cam_trans_md)
        joint_proj_e2, joint_cam_e2, mesh_cam_e2 = \
            self.get_coord(mano_pose_e2[:,:3], mano_pose_e2[:,3:], mano_shape_e2, cam_trans_e2)
    
        if mode == 'train':
            loss = {}
            
            # losses on middle hand; we do not have to consider "order"
            loss['joint_img'] = self.opt_loss['lambda_joint_img'] * self.coord_loss(joint_img_md, targets['joint_img'], meta_info['joint_trunc'], meta_info['is_3D'])
            loss['joint_proj'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj_md, targets['joint_img'][:,:,:2], meta_info['joint_trunc'])
            loss['joint_cam'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam_md, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None])
            loss['mano_joint_cam'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam_md, targets['mano_joint_cam'], meta_info['mano_joint_valid'])
            loss['mano_pose'] = self.param_loss(mano_pose_md, targets['mano_pose'], meta_info['mano_pose_valid'])
            loss['mano_shape'] = self.param_loss(mano_shape_md, targets['mano_shape'], meta_info['mano_shape_valid'][:,None])
            
            # losses on hands in both ends
            # 1) temporal order invariant loss
            loss_joint_img_pf, pred_order = self.coord_loss_order_invariant(joint_img_e1, joint_img_e2, targets['joint_img_past'], targets['joint_img_future'],
                                                                            meta_info['joint_trunc_past'], meta_info['joint_trunc_future'], meta_info['is_3D'], return_order=True)
            loss['joint_img_pf'] = self.opt_loss['lambda_joint_img'] * loss_joint_img_pf
            
            # 2) unfolder driven temporal order loss; use predicted order from Unfolder
            joint_proj_p = joint_proj_e1 * pred_order[:,None,None] + joint_proj_e2 * (1-pred_order[:,None,None])
            joint_proj_f = joint_proj_e2 * pred_order[:,None,None] + joint_proj_e1 * (1-pred_order[:,None,None])
            loss['joint_proj_past'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj_p, targets['joint_img_past'][:,:,:2], meta_info['joint_trunc_past'])
            loss['joint_proj_future'] = self.opt_loss['lambda_joint_proj'] * self.coord_loss(joint_proj_f, targets['joint_img_future'][:,:,:2], meta_info['joint_trunc_future'])
            
            joint_cam_p = joint_cam_e1 * pred_order[:,None,None] + joint_cam_e2 * (1-pred_order[:,None,None])
            joint_cam_f = joint_cam_e2 * pred_order[:,None,None] + joint_cam_e1 * (1-pred_order[:,None,None])
            loss['joint_cam_past'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam_p, targets['joint_cam_past'], meta_info['joint_valid_past'])
            loss['joint_cam_future'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam_f, targets['joint_cam_future'], meta_info['joint_valid_future'])
            loss['mano_joint_cam_past'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam_p, targets['mano_joint_cam_past'], meta_info['mano_joint_valid_past'])
            loss['mano_joint_cam_future'] = self.opt_loss['lambda_joint_cam'] * self.coord_loss(joint_cam_f, targets['mano_joint_cam_future'], meta_info['mano_joint_valid_future'])
            
            mano_pose_p = mano_pose_e1 * pred_order[:,None] + mano_pose_e2 * (1-pred_order[:,None])
            mano_pose_f = mano_pose_e2 * pred_order[:,None] + mano_pose_e1 * (1-pred_order[:,None])
            loss['mano_pose_past'] = self.param_loss(mano_pose_p, targets['mano_pose_past'], meta_info['mano_pose_valid_past'])
            loss['mano_pose_future'] = self.param_loss(mano_pose_f, targets['mano_pose_future'], meta_info['mano_pose_valid_future'])
            
            mano_shape_p = mano_shape_e1 * pred_order[:,None] + mano_shape_e2 * (1-pred_order[:,None])
            mano_shape_f = mano_shape_e2 * pred_order[:,None] + mano_shape_e1 * (1-pred_order[:,None])
            loss['mano_shape_past'] = self.param_loss(mano_shape_p, targets['mano_shape_past'], meta_info['mano_shape_valid_past'][:,None])
            loss['mano_shape_future'] = self.param_loss(mano_shape_f, targets['mano_shape_future'], meta_info['mano_shape_valid_future'][:,None])
        
            return loss
            
        else:
            out = {}
            out['img'] = inputs['img']
            
            # our model predictions
            # when evaluating hands in both ends, MPJPE will be calculated in order of minimizing the value
            out['mano_mesh_cam'] = mesh_cam_md
            out['mano_mesh_cam_past'] = mesh_cam_e1
            out['mano_mesh_cam_future'] = mesh_cam_e2

            # ground-truth mano coordinate
            with torch.no_grad():
                batch_size = inputs['img'].shape[0]
                mesh_coord_cam_gt = torch.zeros((batch_size, mano.vertex_num, 3)).cuda()

                pose_param_right = targets['mano_pose'][meta_info['hand_type']==1]
                shape_param_right = targets['mano_shape'][meta_info['hand_type']==1]
                
                if pose_param_right.shape[0] != 0:
                    mano_output_right_gt = self.mano_layer_right(global_orient=pose_param_right[:,:3], hand_pose=pose_param_right[:,3:], betas=shape_param_right)
                    mesh_coord_cam_right_gt = mano_output_right_gt.vertices
                    mesh_coord_cam_right_gt -= mano_output_right_gt.joints[:,0,:][:,None,:]
                    mesh_coord_cam_gt[meta_info['hand_type']==1] = mesh_coord_cam_right_gt

                pose_param_left = targets['mano_pose'][meta_info['hand_type']==0]
                shape_param_left = targets['mano_shape'][meta_info['hand_type']==0]
                
                if pose_param_left.shape[0] != 0:
                    mano_output_left_gt = self.mano_layer_left(global_orient=pose_param_left[:,:3], hand_pose=pose_param_left[:,3:], betas=shape_param_left)
                    mesh_coord_cam_left_gt = mano_output_left_gt.vertices
                    mesh_coord_cam_left_gt -= mano_output_left_gt.joints[:,0,:][:,None,:]
                    mesh_coord_cam_gt[meta_info['hand_type']==0] = mesh_coord_cam_left_gt
                    
                out['mesh_coord_cam_gt'] = mesh_coord_cam_gt

            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'mano_mesh_cam' in targets:
                out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
                
            return out
        
    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.opt_params['focal'][0] * self.opt_params['focal'][1] * self.opt_params['camera_3d_size'] * \
            self.opt_params['camera_3d_size'] / (self.opt_params['input_img_shape'][0] * self.opt_params['input_img_shape'][1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        
        return cam_trans

    def get_coord(self, root_pose, hand_pose, shape, cam_trans):
        batch_size = root_pose.shape[0]
        output = self.mano_layer_right(global_orient=root_pose, hand_pose=hand_pose, betas=shape)
        
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(mano.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
        
        # project 3D coordinates to 2D space
        x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * \
            self.opt_params['focal'][0] + self.opt_params['princpt'][0]
        y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * \
            self.opt_params['focal'][1] + self.opt_params['princpt'][1]
        x = x / self.opt_params['input_img_shape'][1] * self.opt_params['output_hm_shape'][2]
        y = y / self.opt_params['input_img_shape'][0] * self.opt_params['output_hm_shape'][1]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,mano.root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam

        # add camera translation for the rendering
        mesh_cam = mesh_cam + cam_trans[:,None,:]
        
        return joint_proj, joint_cam, mesh_cam
   