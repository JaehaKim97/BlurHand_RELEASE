import copy
import json
import numpy as np
import os
import os.path as osp
import torch

from pycocotools.coco import COCO
from utils.MANO import mano
from utils.visualize import save_obj, seq2video
from utils.preprocessing import load_img, get_bbox, process_bbox, augmentation, process_db_coord_pcf, process_human_model_output
from utils.transforms import world2cam, cam2pixel, transform_joint_to_other_db


class InterHand26M(torch.utils.data.Dataset):
    def __init__(self, opt, opt_data, transform, data_split):
        self.opt = opt
        self.opt_params = opt['task_parameters']
        self.transform = transform
        self.data_split = 'train' if data_split == 'train' else 'test'
        
        # path for images and annotations
        self.img_path = opt_data['img_path']
        self.annot_path = opt_data['annot_path']

        # IH26M joint set
        self.joint_set = {'hand': \
                            {'joint_num': 21, # single hand
                            'joints_name': ('Thumb_4', 'Thumb_3', 'Thumb_2', 'Thumb_1',
                                            'Index_4', 'Index_3', 'Index_2', 'Index_1',
                                            'Middle_4', 'Middle_3', 'Middle_2', 'Middle_1',
                                            'Ring_4', 'Ring_3', 'Ring_2', 'Ring_1',
                                            'Pinky_4', 'Pinky_3', 'Pinky_2', 'Pinky_1',
                                            'Wrist'),
                            'flip_pairs': (),
                            'skeleton': ((20,3), (3,2), (2,1), (1,0),
                                         (20,7), (7,6), (6,5), (5,4),
                                         (20,11), (11,10), (10,9), (9,8),
                                         (20,15), (15,14), (14,13), (13,12),
                                         (20,19), (19,18), (18,17), (17,16))
                            }
                        }
        self.joint_set['hand']['joint_type'] = {'right': np.arange(0,self.joint_set['hand']['joint_num']),
                                                'left': np.arange(self.joint_set['hand']['joint_num'],
                                                                  self.joint_set['hand']['joint_num']*2)}
        self.joint_set['hand']['root_joint_idx'] = self.joint_set['hand']['joints_name'].index('Wrist')
        self.datalist = self.load_data()

    def load_data(self):
        # we use BlurHand annotations, to only use InterHand images which correspond to the BlurHand
        # for '*_data.json' files, use pycocotools for fast loading
        db = COCO(osp.join(self.annot_path, self.data_split, 'BlurHand_' + self.data_split + '_data.json'))

        # otherwise, use standard json protocol
        with open(osp.join(self.annot_path, self.data_split, 'BlurHand_' + self.data_split + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)      
        with open(osp.join(self.annot_path, self.data_split, 'BlurHand_' + self.data_split + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.data_split, 'BlurHand_' + self.data_split + '_joint_3d.json')) as f:
            joints = json.load(f)
        
        datalist = []
        for aid in [*db.anns.keys()]:
            ann = db.anns[aid]
            
            # load annotation only if the image_id corresponds to the middle frame of BlurHand
            if not ann['is_middle']:
                continue

            # load each item from annotation
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.img_path, self.data_split, img['file_name']).replace('.png', '.jpg')  # compensate different extension
            capture_id = img['capture']
            cam = img['camera']
            frame_idx = img['frame_idx']

            # camera parameters
            t = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32).reshape(3)
            R = np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32).reshape(3,3)
            t = -np.dot(R,t.reshape(3,1)).reshape(3)  # -Rt -> t
            focal = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32).reshape(2)
            princpt = np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32).reshape(2)
            cam_param = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}
           
            # if root is not valid, root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(-1,1)
            joint_valid[self.joint_set['hand']['joint_type']['right']] *= joint_valid[self.joint_set['hand']['root_joint_idx']]
            joint_valid[self.joint_set['hand']['joint_type']['left']] *= joint_valid[self.joint_set['hand']['joint_num'] + \
                                                                                     self.joint_set['hand']['root_joint_idx']]
            # joint coordinates
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32).reshape(-1,3)
            joint_cam = world2cam(joint_world, R, t)
            joint_cam[np.tile(joint_valid==0, (1,3))] = 1.  # prevent zero division error
            joint_img = cam2pixel(joint_cam, focal, princpt)
            
            # for past(1st) and future(5th) image, we just use annotations from middle frame

            # handle the hand_type; ['right', 'left', 'interacting']
            if ann['hand_type'] == 'right':
                hand_type_list = ('right',)
            elif ann['hand_type'] == 'left':
                hand_type_list = ('left',)
            else:
                hand_type_list = ('right', 'left')

            for hand_type in hand_type_list:
                # no avaiable valid joint
                if np.sum(joint_valid[self.joint_set['hand']['joint_type'][hand_type]]) == 0:
                    continue
                
                # process bbox 
                bbox = get_bbox(joint_img[self.joint_set['hand']['joint_type'][hand_type],:2],
                                joint_valid[self.joint_set['hand']['joint_type'][hand_type],0],
                                extend_ratio=1.2)
                bbox = process_bbox(bbox, img_width, img_height, self.opt_params['input_img_shape'])

                # no avaiable bbox
                if bbox is None:
                    continue
                
                # mano parameters for middle frame
                try:
                    mano_param = mano_params[str(capture_id)][str(frame_idx)][hand_type]
                    if mano_param is not None:
                        mano_param['hand_type'] = hand_type
                except KeyError:
                    mano_param = None

                data = {}
                # curremt (middle) frame related
                data['img_path'] = img_path
                data['img_shape'] = (img_height, img_width)
                data['bbox'] = bbox
                data['joint_img'] = joint_img[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_cam'] = joint_cam[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_valid'] = joint_valid[self.joint_set['hand']['joint_type'][hand_type],:]
                data['cam_param'] = cam_param
                data['mano_param'] = mano_param
                data['hand_type'] = hand_type
                data['orig_hand_type'] = ann['hand_type']

                # past frame related, just use value from middle frame
                data['mano_param_past'] = mano_param
                data['joint_img_past'] = joint_img[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_cam_past'] = joint_cam[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_valid_past'] = joint_valid[self.joint_set['hand']['joint_type'][hand_type],:]

                # future frame related, just use value from middle frame
                data['mano_param_future'] = mano_param
                data['joint_img_future'] = joint_img[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_cam_future'] = joint_cam[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_valid_future'] = joint_valid[self.joint_set['hand']['joint_type'][hand_type],:]

                datalist.append(data)

        print("Total number of sample in InterHand2.6M: {}".format(len(datalist)))
        return datalist

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, hand_type = data['img_path'], data['img_shape'], data['bbox'], data['hand_type']
        
        img = load_img(img_path)
        data['cam_param']['t'] /= 1000 # milimeter to meter
        
        # enforce flip when left hand to make it right hand
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split,
                                                                     self.opt_params['input_img_shape'],
                                                                     enforce_flip=(hand_type=='left'))
        img = self.transform(img.astype(np.float32)) / 255.

        if self.data_split != 'train':
            do_flip = False

        # mano parameters for middle
        mano_param = data['mano_param']
        if mano_param is not None:
            mano_joint_img, mano_joint_cam, mano_joint_trunc, mano_pose, mano_shape, mano_mesh_cam_orig = \
                process_human_model_output(mano_param, data['cam_param'], do_flip,
                                           img_shape, img2bb_trans, rot, 'mano',
                                           self.opt_params['input_img_shape'], self.opt_params['output_hm_shape'], self.opt_params['bbox_3d_size'])
            mano_joint_valid = np.ones((mano.joint_num, 1), dtype=np.float32)
            mano_pose_valid = np.ones((mano.orig_joint_num * 3), dtype=np.float32)
            mano_shape_valid = float(True)
        else: # just fill with dummy values
            mano_joint_img = np.zeros((mano.joint_num, 3), dtype=np.float32)
            mano_joint_cam = np.zeros((mano.joint_num, 3), dtype=np.float32)
            mano_joint_trunc = np.zeros((mano.joint_num, 1), dtype=np.float32)
            mano_pose = np.zeros((mano.orig_joint_num * 3), dtype=np.float32) 
            mano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
            mano_joint_valid = np.zeros((mano.joint_num, 1), dtype=np.float32)
            mano_pose_valid = np.zeros((mano.orig_joint_num * 3), dtype=np.float32)
            mano_shape_valid = float(False)

        # prepare data for training
        if self.data_split == 'train':
            joint_cam = data['joint_cam']
            joint_cam_past = data['joint_cam_past']
            joint_cam_future = data['joint_cam_future']            
            
           # root-relative joint in camera coordinates, milimeter to meter.
            joint_cam = (joint_cam - joint_cam[self.joint_set['hand']['root_joint_idx'],None,:]) / 1000 
            joint_cam_past = (joint_cam_past - joint_cam_past[self.joint_set['hand']['root_joint_idx'],None,:]) / 1000
            joint_cam_future = (joint_cam_future - joint_cam_future[self.joint_set['hand']['root_joint_idx'],None,:]) / 1000

            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1)            
            joint_img_past = data['joint_img_past']
            joint_img_past = np.concatenate((joint_img_past[:,:2], joint_cam_past[:,2:]),1)
            joint_img_future = data['joint_img_future']
            joint_img_future = np.concatenate((joint_img_future[:,:2], joint_cam_future[:,2:]),1)

            joint_img_past, joint_img, joint_img_future, joint_cam_past, joint_cam, joint_cam_future, joint_valid_past, joint_valid, joint_valid_future, joint_trunc_past, joint_trunc, joint_trunc_future = \
            process_db_coord_pcf(joint_img_past, joint_img, joint_img_future,
                                 joint_cam_past, joint_cam, joint_cam_future,
                                 data['joint_valid_past'], data['joint_valid'], data['joint_valid_future'],
                                 do_flip, img_shape, self.joint_set['hand']['flip_pairs'], img2bb_trans,
                                 rot, self.joint_set['hand']['joints_name'], mano.joints_name,
                                 self.opt_params['input_img_shape'], self.opt_params['output_hm_shape'], self.opt_params['bbox_3d_size'])

            inputs = {'img': img, 'img_path': img_path}
            targets = {'joint_img': joint_img, 'joint_img_past': joint_img_past, 'joint_img_future': joint_img_future,
                       'joint_cam_past': joint_cam_past, 'joint_cam': joint_cam, 'joint_cam_future': joint_cam_future,
                       'mano_joint_img': mano_joint_img,
                       'mano_joint_cam_past': mano_joint_cam, 'mano_joint_cam': mano_joint_cam, 'mano_joint_cam_future': mano_joint_cam,
                       'mano_pose_past': mano_pose, 'mano_pose': mano_pose, 'mano_pose_future': mano_pose,
                       'mano_shape': mano_shape, 'mano_shape_past': mano_shape, 'mano_shape_future': mano_shape}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 
                         'joint_valid_past': joint_valid_past, 'joint_trunc_past': joint_trunc_past, 
                         'joint_valid_future': joint_valid_future, 'joint_trunc_future': joint_trunc_future, 
                         'mano_joint_trunc': mano_joint_trunc, 
                         'mano_joint_valid_past': mano_joint_valid, 'mano_joint_valid': mano_joint_valid, 'mano_joint_valid_future': mano_joint_valid,
                         'mano_pose_valid': mano_pose_valid, 'mano_pose_valid_past': mano_pose_valid, 'mano_pose_valid_future': mano_pose_valid,
                         'mano_shape_valid': mano_shape_valid, 'mano_shape_valid_past': mano_shape_valid, 'mano_shape_valid_future': mano_shape_valid,
                         'is_3D': float(True)}
        
        # prepare data for testing
        else:
            inputs = {'img': img, 'img_path': img_path}
            targets = {'mano_pose': mano_pose, 'mano_shape': mano_shape}
            meta_info = {'bb2img_trans': bb2img_trans, 'hand_type': 1. if data['hand_type']=='right' else 0.}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe_current': [[] for _ in range(self.joint_set['hand']['joint_num'])],
                       'mpvpe': []}

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            # ground-truth
            mesh_gt = out['mesh_coord_cam_gt']
            
            joint_cam_gt = annot['joint_cam'] / 1000
            joint_cam_gt -= joint_cam_gt[self.joint_set['hand']['root_joint_idx']]
            joint_valid = annot['joint_valid']

            # prediction
            mesh_out = out['mano_mesh_cam']

            joint_cam_out = np.dot(mano.joint_regressor, mesh_out)
            joint_cam_out = transform_joint_to_other_db(joint_cam_out, mano.joints_name, self.joint_set['hand']['joints_name'])
            mesh_out -= joint_cam_out[self.joint_set['hand']['root_joint_idx']]
            joint_cam_out -= joint_cam_out[self.joint_set['hand']['root_joint_idx']]
            
            if annot['hand_type'] == 'left':
                joint_cam_out[:,0] *= -1
                mesh_out[:,0] *= -1
            
            # save obj files
            basename, _ = osp.splitext(osp.basename(annot['img_path']))
            if self.opt['test'].get('save_obj', False):
                obj_dir = osp.join('experiments', self.opt['name'], 'results', 'obj', *annot['img_path'].split('/')[-5:-1])
                os.makedirs(obj_dir, exist_ok=True)
                
                save_obj(mesh_out*np.array([1,-1,-1]), mano.face['right'], osp.join(obj_dir, f'{basename}.obj'))
                
            # calculating MPVPE
            if annot['mano_param'] is not None:
                eval_result['mpvpe'].append((np.sqrt(np.sum(((mesh_out - mesh_gt)**2), 1)) * 1000).mean())

            # calculating MPJPE
            for j in range(self.joint_set['hand']['joint_num']):
                if joint_valid[j]:
                    eval_result['mpjpe_current'][j].append(np.sqrt(np.sum((joint_cam_out[j] - joint_cam_gt[j])**2)) * 1000)

        return eval_result
    
    def print_eval_result(self, logger, eval_result, evaluate_both_ends=False):

        for k, v in eval_result.items():
            if k != 'mpvpe':
                for j in range(self.joint_set['hand']['joint_num']):
                    v[j] = np.mean(np.stack(v[j]))
            eval_result[k] = v
        logger.info('MPJPE @ CURRENT: %.2f mm' % np.mean(eval_result['mpjpe_current']))
        logger.info('MPVPE @ CURRENT: %.2f mm' % np.mean(eval_result['mpvpe']))
        
        return
    
    def __len__(self):
        return len(self.datalist)
