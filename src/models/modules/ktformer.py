import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

class KTFormer(nn.Module):
    def __init__(self, opt_params, in_chans=512, embed_dim=512,
                 num_blocks=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_joints = opt_params['num_joints']

        self.patch_embed = nn.Linear(in_chans, embed_dim)
        self.pos_embed_t = nn.Parameter(torch.randn(1, 3, 1, embed_dim))  # time direction
        self.pos_embed_j = nn.Parameter(torch.randn(1, 1, self.num_joints, embed_dim))  # joint direction
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(num_blocks)])

    def forward(self, feat_joint_e1, feat_joint_md, feat_joint_e2):
        # concat the joint features
        x = torch.cat((feat_joint_e1, feat_joint_md, feat_joint_e2), dim=1)
        
        # forwarding transformer block
        x = self.patch_embed(x)
        x = x + (self.pos_embed_t + self.pos_embed_j).view(1,-1,512)
        for blk in self.blocks:
            x = blk(x)
            
        # channel-wise dividing operation
        feat_joint_e1, feat_joint_md, feat_joint_e2 = torch.split(x, split_size_or_sections=self.num_joints, dim=1)
        
        return feat_joint_e1, feat_joint_md, feat_joint_e2
