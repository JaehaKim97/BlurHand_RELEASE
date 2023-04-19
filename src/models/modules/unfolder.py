import torch
import torch.nn as nn

from models.modules.layer_utils import make_conv_layers, make_deconv_layers
from utils.transforms import soft_argmax_3d, sample_joint_features


class Unfolder(nn.Module):
    def __init__(self, opt_params, **kwargs):
        super().__init__()
        self.num_joints = opt_params['num_joints']
        self.output_hm_shape = opt_params['output_hm_shape']
        
        # make separate decoder for each time step
        self.decoder_e1 = Decoder(self.num_joints, self.output_hm_shape, **kwargs)  # end1
        self.decoder_md = Decoder(self.num_joints, self.output_hm_shape, **kwargs)  # middle
        self.decoder_e2 = Decoder(self.num_joints, self.output_hm_shape, **kwargs)  # end2
        
    def forward(self, feat_blur, feat_pyramid):
        # extract feature using each separate decoder
        feat_e1, pred_heatmap_e1 = self.decoder_e1(feat_blur, feat_pyramid)
        feat_md, pred_heatmap_md = self.decoder_md(feat_blur, feat_pyramid)
        feat_e2, pred_heatmap_e2 = self.decoder_e2(feat_blur, feat_pyramid)

        # obtain 3d joint coordinates using soft-argmax
        joint_img_e1 = soft_argmax_3d(pred_heatmap_e1.reshape(-1, self.num_joints, *self.output_hm_shape))
        joint_img_md = soft_argmax_3d(pred_heatmap_md.reshape(-1, self.num_joints, *self.output_hm_shape))
        joint_img_e2 = soft_argmax_3d(pred_heatmap_e2.reshape(-1, self.num_joints, *self.output_hm_shape))
        
        # grid sampling on feature
        feat_joint_e1 = sample_joint_features(feat_e1, joint_img_e1)  # B J 512
        feat_joint_md = sample_joint_features(feat_md, joint_img_md)  # B J 512
        feat_joint_e2 = sample_joint_features(feat_e2, joint_img_e2)  # B J 512    

        return feat_joint_e1, feat_joint_md, feat_joint_e2, joint_img_e1, joint_img_md, joint_img_e2


class Decoder(nn.Module):
    def __init__(self, num_joints, output_hm_shape, in_chans=2048, out_chans=512):
        super().__init__()
        self.num_joints = num_joints
        self.output_hm_shape = output_hm_shape 
        
        # upsample the spatial resolution using feature pyramid
        ic, oc = in_chans, out_chans
        self.deconv1 = make_deconv_layers([ic, ic//4])
        self.skip1 = make_conv_layers([ic//2, ic//2], kernel=1, padding=0)
        self.conv1 = make_conv_layers([ic//4 + ic//2, ic//2])

        self.deconv2 = make_deconv_layers([ic//2, ic//8])
        self.skip2 = make_conv_layers([ic//4, ic//4], kernel=1, padding=0)
        self.conv2 = make_conv_layers([ic//8 + ic//4, oc])
    
        embedding_chans = num_joints * output_hm_shape[0]
        self.conv_proj = make_conv_layers([oc, embedding_chans], kernel=1, padding=0, bnrelu_final=False)

    def forward(self, feat_blur, feat_pyramid):        
        # feat_blur                : B x ic x 8 x 8
        # feat_pyramid['stride16'] : B x (ic//2) x 16 x 16
        # feat_pyramid['stride8']  : B x (ic//4) x 32 x 32
        
        x = self.deconv1(feat_blur)
        y = self.skip1(feat_pyramid['stride16'])
        x = torch.cat((x, y), 1)
        x = self.conv1(x)

        x = self.deconv2(x)
        y = self.skip2(feat_pyramid['stride8'])
        x = torch.cat((x, y), 1)
        x = self.conv2(x)

        # project into d * J dimension and reshape feature as heatmap
        heatmap = self.conv_proj(x).reshape(-1, self.num_joints, *self.output_hm_shape)

        return x, heatmap
