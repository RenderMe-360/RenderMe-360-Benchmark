# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import copy
import random
import json
import cv2
import torch
import imageio
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from kornia.geometry.conversions import convert_points_to_homogeneous

def load_json(fname):
	with open(fname, 'r') as fp:
		data = json.load(fp)
	IDs = sorted([int(k.split('_')[0]) for k in data.keys() if k[-2:] == '_K'])
	Ks = []; w2cs= []; dists = []; camIDs = []
	for ID in IDs:
		K  = np.reshape(data['%d_K' % ID], [3,3])
		dist=np.reshape(data.get('%d_distortion'%ID,[]), -1)
		dist=None if len(dist)==0 else np.concatenate([dist,np.zeros([max(5-len(dist),0)])])
		w2c =np.reshape(data.get('%d_Rt' % ID, np.identity(4)), [-1,4])
		h = int(data['%d_height' % ID])
		w = int(data['%d_width'  % ID])
		valid = data.get('%d_valid' % ID, False)
		if valid:
			Ks.append(K)
			w2cs.append(w2c[:3,:4])
			dists.append(dist)
			camIDs.append(ID)
	return	Ks, w2cs, dists, camIDs

def load_json_dict(fname):
	with open(fname, 'r') as fp:
		data = json.load(fp)
	IDs = sorted([int(k.split('_')[0]) for k in data.keys() if k[-2:] == '_K'])
	cameras = {}
	for ID in IDs:
		K  = np.reshape(data['%d_K' % ID], [3,3])
		dist=np.reshape(data.get('%d_distortion'%ID,[]), -1)
		dist=None if len(dist)==0 else np.concatenate([dist,np.zeros([max(5-len(dist),0)])])
		w2c =np.reshape(data.get('%d_Rt' % ID, np.identity(4)), [-1,4])
		h = int(data['%d_height' % ID])
		w = int(data['%d_width'  % ID])
		valid = data.get('%d_valid' % ID, False)
		if valid:
			cameras[str(ID)] = {
                "K": K,
                "dist": dist,
                "w2c": w2c[:3,:4]
            }
	return	cameras


class FacescapeDataset(data.Dataset):
    ''' This data loader loads the Zju-MoCap dataset (CVPR'21). '''
    def __init__(self, data_root, split, **kwargs):
        super(FacescapeDataset, self).__init__()

        self.range_min = int(kwargs.get('range_min', 0))
        self.range_max = int(kwargs.get('range_max', 68))
        self.split = split  # 'train'
        path_npy = '../facescape/facescape_train.npy'  # TODO:facescape/facescape_train.npy
        meta = np.load(path_npy, allow_pickle=True).item()
        self.setting = 'setting_2'

        self.data_root = data_root
        self.max_len = kwargs.get('max_len', -1)
        self.origin_H, self.origin_W = 2048, 2448
        self.clip = int((self.origin_W - self.origin_H) / 2)
        
        run_mode = split
        self.image2tensor = transforms.Compose([transforms.ToTensor(), ])
        self.ratio = 0.25
        self.skip = 2

        self.cams = {}
        self.ims = []
        self.mats = []
        self.cam_inds = []
        self.num_frames = []
        self.yaws = []
        self.pitchs = []
        self.start_end = {}
        self.info_val_views = [] # for testing from facescape_test.npy
        self.kpt_mapping = list(range(0,31)) + list(range(51,71))

        pid = kwargs.get('pid', -1)
        self.test_split = 'random_train/random_test/unseen_id'

        test_info = np.load('../facescape/facescape_test.npy', allow_pickle=True).item() # TODO:facescape/facescape_test.npy

        if self.setting == 'setting_2' and 'val' in self.split: 
            info_val_views = test_info[self.test_split]
            human_list = list(info_val_views.keys())
        elif self.split == 'train':
            human_list = [int(h) for h in meta.keys() if int(h) < 300]
        else:
            human_list = [int(h) for h in meta.keys() if 300 <= int(h) < 360]

        if self.split in ['test', 'val_novel_pose', 'val_unseen_id', 'val']:
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        # print(f"human_list: {human_list}")
        from tqdm import tqdm
        for idx in tqdm(range(len(human_list))):
            human = str(human_list[idx])
            if human not in meta.keys():
                continue
            
            exps = meta[human].keys()
            for exp in exps:

                ann_file = os.path.join(self.data_root, "images_aligned", human, exp, 'params.json')
                if not os.path.exists(ann_file):
                    print(f"path not exist: {ann_file}")
                    continue

                cameras = load_json_dict(ann_file)

                if human not in self.cams.keys():
                     self.cams[human] = {}
                self.cams[human][exp] = cameras

                data_root = os.path.join(self.data_root, 'images_aligned', human, exp)
                mask_root = os.path.join(self.data_root, 'matting_aligned', human, exp)
                kpt_root = os.path.join(self.data_root, 'LMK3D/lmk3ds', human, f"{exp}.npy")
                if os.path.exists(data_root) and os.path.exists(kpt_root) and os.path.exists(mask_root):

                    test_views = meta[human][exp]['test_views']
                    for i, test_view in enumerate(test_views):
                        if self.split == 'train':
                            path_img = os.path.join(data_root, f'{test_view}.jpg')
                            path_matting = os.path.join(mask_root, f'{test_view}.png')
                            if (not os.path.exists(path_img)) or (not os.path.exists(path_matting)):
                                continue
                            self.ims.append(path_img)
                            self.mats.append(path_matting)
                            self.num_frames.append(1)
                            self.cam_inds.append(test_views)
                            self.yaws.append(meta[human][exp]['yaw_views'])
                            self.pitchs.append(meta[human][exp]['pitch_views'])
                        
                        # append views from facescape_test.npy
                        elif self.setting == 'setting_2' and 'val' in self.split:
                            # facescape_test.npy only sample part of the test data (expression & views)
                            if exp not in info_val_views[human].keys() or \
                                test_view not in info_val_views[human][exp].keys():
                                continue
                            path_img = os.path.join(data_root, f'{test_view}.jpg')
                            self.ims.append(path_img)
                            path_matting = os.path.join(mask_root, f'{test_view}.png')
                            self.mats.append(path_matting)
                            self.info_val_views.append([test_view] + info_val_views[human][exp][test_view])
        
        self.num_humans = len(human_list)
        print(f"num of target frames set: {len(self.ims)}")

    @classmethod
    def from_config(cls, dataset_cfg, data_split, cfg):
        ''' Creates an instance of the dataset.

        Args:
            dataset_cfg (dict): input configuration.
            data_split (str): data split (`train` or `val`).
        '''
        assert data_split in ['train', 'val_novel_pose', 'val_unseen_id', 'test', 'val']

        dataset_cfg = copy.deepcopy(dataset_cfg)
        dataset_cfg['is_train'] = data_split == 'train'
        # if f'{data_split}_cfg' in dataset_cfg:
            # dataset_cfg.update(dataset_cfg[f'{data_split}_cfg'])
        # if dataset_cfg['is_train']:
        dataset = cls(split=data_split, **dataset_cfg)

        return dataset
    
    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        # sample a frame for training
        tar_img_path = self.ims[index]
        tar_mat_path = self.mats[index]
        # print(f"tar_img_path: {tar_img_path}")
        tar_data_info = tar_img_path.split('/')
        exp = tar_data_info[-2]
        human = tar_data_info[-3]
        tar_view = int(tar_data_info[-1][:-4])

        if self.split == 'train':
            test_views = self.cam_inds[index]
            yaws = self.yaws[index]
            pitchs = self.pitchs[index]
            seq_num = self.num_frames[index]
            while True:
                src_view1 = random.choice(yaws[yaws.index(tar_view)-10 if yaws.index(tar_view)-10>=0 else 0:yaws.index(tar_view)])
                if os.path.exists(tar_img_path.replace(f'{tar_view}.jpg', f'{src_view1}.jpg')) and \
                    os.path.exists(tar_mat_path.replace(f'{tar_view}.png', f'{src_view1}.png')): break

            while True:
                src_view2 = random.choice(pitchs[pitchs.index(tar_view)-8 if pitchs.index(tar_view)-8>=0 else 0:pitchs.index(tar_view)+8 if pitchs.index(tar_view)+8 < len(pitchs) else -1])
                if os.path.exists(tar_img_path.replace(f'{tar_view}.jpg', f'{src_view2}.jpg')) and \
                    os.path.exists(tar_mat_path.replace(f'{tar_view}.png', f'{src_view2}.png')): break

            while True:
                src_view3 = random.choice(yaws[yaws.index(tar_view)+1:yaws.index(tar_view)+11 if yaws.index(tar_view)+11 <len(yaws) else -1])                                
                if os.path.exists(tar_img_path.replace(f'{tar_view}.jpg', f'{src_view3}.jpg')) and \
                    os.path.exists(tar_mat_path.replace(f'{tar_view}.png', f'{src_view3}.png')): break


            input_view = [tar_view] + [src_view1, src_view2, src_view3]
        else:
            input_view = self.info_val_views[index]

        tar_view_ind = input_view[0]
        input_imgs, input_msks, input_K, input_Rt = [], [], [], []
        for ii, idx in enumerate(input_view):
            input_img_path = tar_img_path.replace(f'{tar_view}.jpg', f'{idx}.jpg')
            maskpath = tar_mat_path.replace(f'{tar_view}.png', f'{idx}.png')
            input_msk = (imageio.imread(maskpath) > 127).astype(np.uint8)[:, self.clip:-self.clip]

            # load data
            in_K, in_D = np.array(self.cams[human][exp][str(idx)]['K']).astype(np.float32), np.array(self.cams[human][exp][str(idx)]['dist']).astype(np.float32)
            in_Rt = np.array(self.cams[human][exp][str(idx)]['w2c']).astype(np.float32)
            input_img = imageio.imread(input_img_path).astype(np.float32)[:, self.clip:-self.clip] / 255.

            # resize images
            H, W = int(input_img.shape[0] * self.ratio), int(input_img.shape[1] * self.ratio)
            input_img, input_msk = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)

            # apply foreground mask
            if ii == 0:
                input_img[input_msk == 0] = 1
            else:
                input_img[input_msk == 0] = 0
            input_msk = (input_msk != 0)  # bool mask : foreground (True) background (False)

            input_msk = input_msk.astype(np.uint8) * 255

            # [0,1]
            input_img = self.image2tensor(input_img)
            input_msk = self.image2tensor(input_msk).bool()
            in_K[0,2] = in_K[0,2] - float(self.clip)
            in_K[:2] = in_K[:2] * self.ratio

            # append data
            input_imgs.append(input_img)
            input_msks.append(input_msk)
            input_K.append(torch.from_numpy(in_K))
            input_Rt.append(torch.from_numpy(in_Rt))

        frame_index=0

        joints_path = os.path.join(self.data_root, 'LMK3D/lmk3ds', human, f"{exp}.npy")
        xyz_joints = np.load(joints_path).astype(np.float32)
        kpt51 = np.array(xyz_joints[self.kpt_mapping]).astype(np.float32)

        human_idx = 0
        if self.split in ['test', 'val_novel_pose', 'val_unseen_id']:
            human_idx = self.human_idx_name[human]
        
        ret = {
            'images': torch.stack(input_imgs),
            'images_masks': torch.stack(input_msks),
            'K': torch.stack(input_K),
            'Rt': torch.stack(input_Rt),
            'kpt3d': torch.from_numpy(kpt51),
            'i': frame_index,
            'human_idx': human_idx,
            'sessision': human,
            'frame_index': frame_index,
            'human': human,
            'exp': exp,
            'cam_ind': input_view[0],
            'seq_num': 1,
            "index": {
                "camera": "cam", 
                "segment": 'facescape', 
                "tar_cam_id": tar_view_ind,
                "frame": f"{human}_{exp}_{tar_view_ind}", 
                "ds_idx": idx
            },
        }

        bounds = self.load_human_bounds()
        ret['mask_at_box'] = self.get_mask_at_box(
            bounds,
            input_K[0].numpy(),
            input_Rt[0][:3, :3].numpy(),
            input_Rt[0][:3, -1].numpy(),
            H, W)
        ret['bounds'] = bounds
        ret['mask_at_box'] = ret['mask_at_box'].reshape((H, W))
        '''
        # DEBUG
        print(f"images: {ret['images'].shape} {type(ret['images'])}")
        print(f"images_masks: {ret['images_masks'].shape} {type(ret['images_masks'])}")
        print(f"K: {ret['K'].shape} {type(ret['K'])}")
        print(f"Rt: {ret['Rt'].shape} {type(ret['Rt'])}")
        print(f"kpt3d: {ret['kpt3d'].shape} {type(ret['kpt3d'])}")
        print(f"i: {ret['i']} {type(ret['i'])}")
        print(f"human_idx: {ret['human_idx']} {type(ret['human_idx'])}")
        print(f"sessision: {ret['sessision']} {type(ret['sessision'])}")
        print(f"frame_index: {ret['frame_index']} {type(ret['frame_index'])}")
        print(f"human: {ret['human']} {type(ret['human'])}")
        print(f"exp: {ret['exp']} {type(ret['exp'])}")
        print(f"cam_ind: {ret['cam_ind']} {type(ret['cam_ind'])}")
        print(f"seq_num: {ret['seq_num']} {type(ret['seq_num'])}")
        print(f"index: {ret['index']} {type(ret['index'])}")
        print(f"mask_at_box: {ret['mask_at_box'].shape} {type(ret['mask_at_box'])}")
        print(f"bounds: {ret['bounds'].shape} {type(ret['bounds'])}")
        print(f"mask_at_box: {ret['mask_at_box'].shape} {type(ret['mask_at_box'])}")
        import pdb;pdb.set_trace()
        '''


        return ret

    def get_length(self):
        return self.__len__()

    def __len__(self):
        if self.max_len == -1:
            return len(self.ims)
        else:
            return min(len(self.ims), self.max_len)

    def load_human_bounds(self):
        min_xyz = np.array([-0.18, -0.62, -0.13]).astype(np.float32)
        max_xyz = np.array([ 0.05, -0.29,  0.10]).astype(np.float32)
        volume = max_xyz - min_xyz
        max_xyz = max_xyz + volume * 0.5
        min_xyz = min_xyz - volume * 0.5
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        return bounds

    @staticmethod
    def get_mask_at_box(bounds, K, R, T, H, W):
        ray_o, ray_d = FacescapeDataset.get_rays(H, W, K, R, T)

        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = FacescapeDataset.get_near_far(bounds, ray_o, ray_d)
        return mask_at_box.reshape((H, W))

    @staticmethod
    def get_rays(H, W, K, R, T):
        rays_o = -np.dot(R.T, T).ravel()

        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32), indexing='xy')

        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_world = np.dot(pixel_camera - T.ravel(), R)
        rays_d = pixel_world - rays_o[None, None]
        rays_o = np.broadcast_to(rays_o, rays_d.shape)

        return rays_o, rays_d

    @staticmethod
    def get_near_far(bounds, ray_o, ray_d, boffset=(-0.01, 0.01)):
        """calculate intersections with 3d bounding box"""
        bounds = bounds + np.array([boffset[0], boffset[1]])[:, None]
        nominator = bounds[None] - ray_o[:, None]
        # calculate the step of intersections at six planes of the 3d bounding box
        ray_d[np.abs(ray_d) < 1e-5] = 1e-5
        d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
        # calculate the six interections
        p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
        # calculate the intersections located at the 3d bounding box
        min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
        eps = 1e-6
        p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                        (p_intersect[..., 0] <= (max_x + eps)) * \
                        (p_intersect[..., 1] >= (min_y - eps)) * \
                        (p_intersect[..., 1] <= (max_y + eps)) * \
                        (p_intersect[..., 2] >= (min_z - eps)) * \
                        (p_intersect[..., 2] <= (max_z + eps))
        # obtain the intersections of rays which intersect exactly twice
        mask_at_box = p_mask_at_box.sum(-1) == 2
        p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
            -1, 2, 3)

        # calculate the step of intersections
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        norm_ray = np.linalg.norm(ray_d, axis=1)
        d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
        d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
        near = np.minimum(d0, d1)
        far = np.maximum(d0, d1)

        return near, far, mask_at_box

def draw_keypoints(img, kpts, color=(255, 0, 0), size=3):
    for i in range(kpts.shape[0]):
        kp2 = kpts[i].tolist()
        kp2 = [int(kp2[0]), int(kp2[1])]
        img = cv2.circle(img, kp2, 2, color, size)
    return img

class FacescapeTestDatset(FacescapeDataset):
    def __init__(self, data_root, split, sample_frame=30, sample_camera=1, **kwargs):
        super().__init__(data_root, split, **kwargs)

        # load im list
        self.sc_factor = 1.0

        if split == 'val':
            self.ims = self.ims[:len(self.test_views)][::3]
            self.cam_inds = self.cam_inds[:len(self.test_views)][::3]
        else:
            num_testviews = len(self.test_views)
            self.ims = self.ims[::num_testviews]
            self.cam_inds = self.cam_inds[::num_testviews]

        # print(self.ims)
