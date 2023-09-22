# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import copy
import random

import cv2
import torch
import imageio
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from kornia.geometry.conversions import convert_points_to_homogeneous 

def load_krt(anno_file):
    """Load KRT file containing intrinsic and extrinsic parameters for cameras.
    
    KRT file is a text file with 1 or more entries like:
        <camera name> <image width (pixels)> <image height (pixels)>
        <f_x> 0 <c_x>
        0 <f_y> <c_y>
        0 0 1
        <k_1> <k_2> <p_1> <p_2> <k_3>
        <r_11> <r_12> <r_13> <t_x>
        <r_21> <r_22> <r_23> <t_y>
        <r_31> <r_32> <r_33> <t_z>
        [blank line]

    The parameters are in OpenCV format described here:
        https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

    Note that the code assumes undistorted images as input, so the distortion
    coefficients are ignored."""
    cameras = {}

    with open(anno_file, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            namesplit = name.split()
            if len(namesplit) > 1:
                name, width, height = namesplit[0], namesplit[1], namesplit[2]
                size = {"size": np.array([width, height])}
            else:
                name = namesplit[0]
                size = {"size": np.array([1334, 2048])}

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name] = {
                    "intrin": np.array(intrin),
                    "dist": np.array(dist),
                    "extrin": np.array(extrin), **size}

    return cameras

class MultifaceDataset(data.Dataset):
    ''' This data loader loads the Zju-MoCap dataset (CVPR'21). '''
    def __init__(self, data_root, split, **kwargs):
        super(MultifaceDataset, self).__init__()

        self.range_min = int(kwargs.get('range_min', 0))
        self.range_max = int(kwargs.get('range_max', 60))
        self.split = split  # 'train'
        # setting_1: fixed source view
        # setting_2: random source view
        self.setting = 'setting_2'
        if self.split == 'train':
            self.test_views = ['400048','400029', '400015','400013','400016','400002','400037']
            self.yaw = ['400028','400004','400017','400026','400027','400064','400023','400013','400037',\
                '400049','400007','400029','400048','400030','400015','400002','400069','400041','400016',\
                '400009','400019','400051','400063','400012','400060','400039']
            self.pitch = ['400027','400051','400069','400049','400026','400063','400064','400019','400015',\
                '400023','400029','400009','400028','400017','400016','400013','400012','400039','400030',\
                '400004','400048','400060','400037','400002','400041','400007']

        else:
            self.test_views = ['400357','400356','400400','400357','400300','400289','400406','400405','400350','400354','400401']
            self.yaw = ['400436', '400298', '400447', '400284', '400448', '400441', '400300', '400421',
            '400365', '400289', '400288', '400404', '400449', '400399', '400377', '400406',
            '400400', '400422', '400268', '400405', '400280', '400362', '400356', '400357',
            '400297', '400442', '400408', '400417', '400401', '400378', '400264', '400350',
            '400358', '400413', '400369', '400371', '400347', '400354', '400293', '400348',
            '400263', '400349', '400428', '400281']
            self.pitch = ['400422', '400268', '400408', '400421', '400264', '400280', '400404', '400369',
            '400417', '400284', '400365', '400347', '400405', '400406', '400436', '400350',
            '400428', '400263', '400399', '400442', '400371', '400447', '400289', '400281',
            '400348', '400357', '400448', '400401', '400354', '400400', '400356', '400300',
            '400358', '400349', '400298', '400297', '400449', '400413', '400441', '400378',
            '400362', '400377', '400288', '400293']

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
        self.cam_inds = []
        self.num_frames = []
        self.start_end = {}
        self.views = []
        self.kpt_mapping = list(range(0,31)) + list(range(51,71))

        pid = kwargs.get('pid', -1)
        sub_train = kwargs.get('renface_sub_train', None)
        self.test_split = 'random_train/random_test/unseen_id' # kwargs.get('test_split')
        test_info = np.load('../multiface/multiface_test.npy', allow_pickle=True).item() # TODO: path to multiface/multiface_test.npy
        if self.setting == 'setting_2' and 'val' in self.split:
            human_info = test_info[self.test_split]
            human_list = list(human_info.keys())
        elif self.split == 'train':
            # human_info = get_human_split(split, pid=pid, sub_train=sub_train)
            human_list = ['m--20171024--0000--002757580--GHS',
                      'm--20180406--0000--8870559--GHS',
                      'm--20180927--0000--7889059--GHS',
                      'm--20180105--0000--002539136--GHS',
                      'm--20180418--0000--2183941--GHS',
                      'm--20181017--0000--002914589--GHS', 
                      'm--20180226--0000--6674443--GHS',
                      'm--20180426--0000--002643814--GHS',
                      'm--20180227--0000--6795937--GHS',
                      'm--20180510--0000--5372021--GHS']
        else:
            human_list = ['m--20190529--1004--5067077--GHS',  
                        'm--20190529--1300--002421669--GHS',  
                        'm--20190828--1318--002645310--GHS']

        # self.human_info = copy.deepcopy(human_info)
        # human_list = list(human_info.keys())
        if self.split == 'train':
            exps = ['E003_Neutral_Eyes_Closed','E004_Relaxed_Mouth_Open','E008_Smile_Mouth_Closed']
        else:
            exps = ['EXP_eye_neutral']

        if self.split in ['test', 'val_novel_pose', 'val_unseen_id', 'val']:
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        for idx in range(len(human_list)):
            human = human_list[idx]
            ann_file = os.path.join(self.data_root, human, 'KRT_align.txt')
            # print(f"ann_file: {ann_file}")
            if os.path.exists(ann_file):

                annots = load_krt(ann_file)

                self.cams[human] = annots
                num_cams = len(self.cams[human].keys())
                test_view = self.test_views
                if self.setting == 'setting_2' and 'val_' in self.split:
                    self.views.extend(human_info[human]['views'])

                for exp in exps:
                    # print(human_info[human]['date'], exp)
                    data_root = os.path.join(self.data_root, human, 'images_gamma_aligned', exp)
                    mask_root = os.path.join(self.data_root, human, 'matting_aligned', exp)
                    kpt_root = os.path.join(self.data_root.replace('multiface', 'multiface_LMK3D'), 'lmk3ds', human, exp)
                    if os.path.exists(data_root) and os.path.exists(kpt_root) and os.path.exists(mask_root):
                        view_tmp = os.listdir(data_root)[0]
                        if len(os.listdir(os.path.join(data_root, view_tmp))) == len(os.listdir(kpt_root)):
                            frames = sorted(os.listdir(os.path.join(data_root, view_tmp)))
                            # frames = frames[-30::2] if self.split == 'train' else frames[-21:-20]
                            frames = frames if self.split == 'train' else frames[-21:-20]
                            ims = np.array([
                                np.array([f'{data_root}/%06d/{frame}' for view in test_view])
                                for frame in frames
                            ]).ravel()

                            cam_inds = np.array([
                                test_view
                                for _ in frames
                            ]).ravel()

                            num_frames = np.array([
                                np.array([len(frames) for view in test_view])
                                for _ in frames
                            ]).ravel()

                            start_idx = len(self.ims)
                            length = len(ims)
                            self.ims.extend(ims)
                            self.cam_inds.extend(cam_inds)
                            self.num_frames.extend(num_frames)
            else:
                self.human_info.pop(human)
                print('poping', human, human_info[human]['date'])
                print('ann', os.path.exists(ann_file))
        # self.nrays = cfg.N_rand
        
        self.num_humans = len(human_list)
        print(sub_train, self.num_humans, len(self.ims))

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
        # else:
        #     dataset = RenFaceTestDatset(split=data_split, **dataset_cfg)
        return dataset
    
    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, tar_index):
        # sample a frame for training
        tar_img_path = self.ims[tar_index]
        # print(f"tar_img_path: {tar_img_path}")
        tar_data_info = tar_img_path.split('/')
        human_exp = tar_data_info[-3]
        human = tar_data_info[-5]
        target_frame = tar_data_info[-1][:-4]
        frame_index = int(target_frame)
        seq_num = self.num_frames[tar_index]

        current_frame = int(target_frame)

        if self.split == 'train' or self.split == 'val':
            all_input_view = copy.deepcopy(self.test_views)

            while True:
                random.shuffle(all_input_view)
                tar_view = all_input_view[0]
                if os.path.exists(tar_img_path % int(tar_view)): break
            while True:
                src_view1 = random.choice(self.yaw[self.yaw.index(tar_view)-10 if self.yaw.index(tar_view)-10>=0 else 0:self.yaw.index(tar_view)])
                if os.path.exists(tar_img_path % int(src_view1)): break
            while True:
                src_view2 = random.choice(self.pitch[self.pitch.index(tar_view)-8 if self.pitch.index(tar_view)-8>=0 else 0:self.pitch.index(tar_view)+8 if self.pitch.index(tar_view)+8 < len(self.pitch) else -1])
                if os.path.exists(tar_img_path % int(src_view2)): break
            while True:
                src_view3 = random.choice(self.yaw[self.yaw.index(tar_view)+1:self.yaw.index(tar_view)+11 if self.yaw.index(tar_view)+11 <len(self.yaw) else -1])                                
                if os.path.exists(tar_img_path % int(src_view3)): break

            input_view = [tar_view] + [src_view1, src_view2, src_view3]
        else:
            input_view = self.views[tar_index]

        tar_view_ind = input_view[0]
        input_imgs, input_msks, input_K, input_Rt = [], [], [], []
        for ii, idx in enumerate(input_view):
            input_img_path = tar_img_path % int(idx)
            maskpath = input_img_path.replace('images_gamma_aligned', 'matting_aligned').replace(f'/{idx}/', f'/{idx}/pha/')
            # print(f"maskpath: {maskpath}")
            input_msk = (imageio.imread(maskpath) > 127).astype(np.uint8)[:, self.clip:-self.clip,:1]
            # if input_msk.sum() == 0: 
            #     return None # NOTE: 跳过mask有问题的数据
            
            # print(f"input_msk: {input_msk.shape}")
            # load data
            in_K, in_D = np.array(self.cams[human][idx]['intrin']).astype(np.float32), np.array(self.cams[human][idx]['dist']).astype(np.float32)
            in_Rt = np.array(self.cams[human][idx]['extrin']).astype(np.float32)
            input_img = imageio.imread(input_img_path).astype(np.float32)[:, self.clip:-self.clip] / 255.
            # input_img, input_msk = cv2.undistort(input_img, in_K, in_D), cv2.undistort(input_msk, in_K, in_D)

            # resize images
            H, W = int(input_img.shape[0] * self.ratio), int(input_img.shape[1] * self.ratio)
            input_img, input_msk = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)

            # apply foreground mask
            if ii == 0:
                input_img[input_msk == 0] = 1
            else:
                input_img[input_msk == 0] = 0
            input_msk = (input_msk != 0)  # bool mask : foreground (True) background (False)

            # apply foreground mask
            # kernel = np.ones((erode_border, erode_border), np.uint8)
            # input_msk = cv2.erode(input_msk.astype(np.uint8) * 255, kernel)
            # kernel = np.ones((erode_border, erode_border), np.uint8)
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

        i = int(frame_index)
        joints_path = os.path.join(self.data_root.replace('multiface', 'multiface_LMK3D'), 'lmk3ds', human, human_exp, tar_data_info[-1].replace('.png', '.npy'))
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
            'i': i,
            'human_idx': human_idx,
            'sessision': human,
            'frame_index': frame_index,
            'human': human,
            'human_exp': human_exp,
            'cam_ind': input_view[0],
            'seq_num': seq_num,
            "index": {"camera": "cam", "segment": 'renderme_test_multiface', "tar_cam_id": tar_view_ind,
                "frame": f"{human_exp}_{frame_index:06d}_{tar_view_ind}", "ds_idx": idx},
        }

        bounds = self.load_human_bounds(human, i)
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
        print(f"human_exp: {ret['human_exp']} {type(ret['human_exp'])}")
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

    def load_human_bounds(self, human, i):
        min_xyz = np.array([-0.18, -0.62, -0.13]).astype(np.float32)
        max_xyz = np.array([ 0.05, -0.29,  0.10]).astype(np.float32)
        volume = max_xyz - min_xyz
        max_xyz = max_xyz + volume * 0.8 # TODO: 0.5
        min_xyz = min_xyz - volume * 0.8
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        return bounds

    @staticmethod
    def get_mask_at_box(bounds, K, R, T, H, W):
        ray_o, ray_d = MultifaceDataset.get_rays(H, W, K, R, T)

        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = MultifaceDataset.get_near_far(bounds, ray_o, ray_d)
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

class MultifaceTestDatset(MultifaceDataset):
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
