import numpy as np
import os
import os.path as osp
import torch

from utils.transforms import transform_joint_to_other_db
from utils.MANO.smplx.smplx.body_models import create

MANO_MODEL_PATH = osp.join(osp.dirname(os.path.abspath(__file__)), 'models')

class MANO(object):
    def __init__(self, flat_hand_mean=False):
        if not os.path.isdir(MANO_MODEL_PATH):
            raise NotImplementedError(f'MANO model pkl files are not properly loacted! SHOULD BE LOCATED in {MANO_MODEL_PATH}')
        
        self.layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
        # self.layer = {'right': smplx.create(MANO_MODEL_PATH, 'mano', is_rhand=True, use_pca=False, flat_hand_mean=flat_hand_mean, **self.layer_arg), 'left': smplx.create(MANO_MODEL_PATH, 'mano', is_rhand=False, use_pca=False, flat_hand_mean=flat_hand_mean, **self.layer_arg)}
        self.layer = {'right': create(MANO_MODEL_PATH, 'mano', is_rhand=True, use_pca=False, flat_hand_mean=flat_hand_mean, **self.layer_arg),
                      'left': create(MANO_MODEL_PATH, 'mano', is_rhand=False, use_pca=False, flat_hand_mean=flat_hand_mean, **self.layer_arg)}
        self.vertex_num = 778
        self.face = {'right': self.layer['right'].faces, 'left': self.layer['left'].faces}
        self.shape_param_dim = 10

        if torch.sum(torch.abs(self.layer['left'].shapedirs[:,0,:] - self.layer['right'].shapedirs[:,0,:])) < 1:
            # print('Fix shapedirs bug of MANO')
            self.layer['left'].shapedirs[:,0,:] *= -1

        # original MANO joint set
        self.orig_joint_num = 16
        self.orig_joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3')
        self.orig_root_joint_idx = self.orig_joints_name.index('Wrist')
        self.orig_flip_pairs = ()
        self.orig_joint_regressor = self.layer['right'].J_regressor.numpy() # same for the right and left hands

        # changed MANO joint set
        self.joint_num = 21 # manually added fingertips
        self.joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
        self.skeleton = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )
        self.root_joint_idx = self.joints_name.index('Wrist')
        self.flip_pairs = ()
        # add fingertips to joint_regressor
        self.joint_regressor = transform_joint_to_other_db(self.orig_joint_regressor, self.orig_joints_name, self.joints_name)
        self.joint_regressor[self.joints_name.index('Thumb_4')] = np.array([1 if i == 745 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Index_4')] = np.array([1 if i == 317 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Middle_4')] = np.array([1 if i == 445 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Ring_4')] = np.array([1 if i == 556 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Pinky_4')] = np.array([1 if i == 673 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)

mano = MANO()
