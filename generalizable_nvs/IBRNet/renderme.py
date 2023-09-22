
import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys, copy, random, cv2
sys.path.append('../')
from .data_utils import random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses



def get_human_split(split, pid=0):
    all_set = np.loadtxt('renderme_all.txt', dtype=str).ravel()
    test_set = np.loadtxt('renderme_test.txt', dtype=str).ravel()
    test20_set = np.loadtxt('renderme_test20.txt', dtype=str).ravel()
    train20_set = np.loadtxt('renderme_train20.txt', dtype=str).ravel()
    if split == 'train':
        ret = {}
        for name in sorted(list(set(all_set)-set(test_set))):
            date, subject = name[:4], name[5:]
            ret[subject] = {'date': date}
        return ret
    elif split == 'val_novel_pose':
        ret = {}
        for name in sorted(train20_set)[pid*2:pid*2+2]:
            date, subject = name[:4], name[5:]
            ret[subject] = {'date': date}
        return ret
    else:
        ret = {}
        for name in sorted(test20_set)[pid*3:pid*3+3]:
            date, subject = name[:4], name[5:]
            ret[subject] = {'date': date}
        return ret

class RenderMeDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.args = args
        self.mode = mode  # train / test / validation
        self.range_min = int(kwargs.get('range_min', 0))
        self.range_max = int(kwargs.get('range_max', 60))

        self.data_root = args.rootdir# mode = [protocal_1,protocal_2,indomain]
        #setting1
        self.test_input_view = [19, 25, 31]
        self.num_source_views = len(self.test_input_view)
        self.test_views =[21,22, 23, 24, 26, 27, 29]

        #setting2:
        self.setting2 = True if 'protocal_1' not in self.data_root else False
        if self.setting2:
            self.test_input_view = []
            self.test_views = list(range(15, 36))

        #setting2-test
        self.test_views = [21, 22, 23, 24, 26, 27, 29]

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []
        

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        self.skip = 2 if self.mode == 'train' else 1
        self.origin_H, self.origin_W = 2048, 2448
        self.clip = int((self.origin_W - self.origin_H) / 2)
        self.ratio = 0.25

        self.cams = {}
        self.ims = []
        self.cam_inds = []
        self.num_frames = []
        self.views = []
        self.start_end = {}
        

        
        if self.setting2 :
            self.test_split = args.eval_scenes[0]#kwargs.get('test_split')
            test_info = np.load('../renderme_test.npy', allow_pickle=True).item()
            human_info = test_info[self.test_split]


            self.human_info = copy.deepcopy(human_info)
            human_list = list(human_info.keys())
        else:
            if len(args.eval_scenes)!=0:
                date,id = args.eval_scenes[0].split('-')
                human_info = {id:{'date':date}}
            else:
                human_info = get_human_split(mode)
            self.human_info = copy.deepcopy(human_info)
            human_list = list(human_info.keys())

            
        
        for idx in range(len(human_list)):
            human = human_list[idx]
            if self.setting2:
                expressions = human_info[human]['exps']
            ann_file = os.path.join(self.data_root, 'annotations/params', expressions[0], 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()

            self.cams[human] = annots['cams']
            test_view = self.test_views
            self.views.extend(human_info[human]['views'])

            for exp in expressions:
                data_root = os.path.join(self.data_root, 'raw',  f'{exp}')
               

                ims =  np.array([f'{data_root}/%02d.jpg' for view in test_view]).ravel()


                self.ims.extend(ims)
    def __len__(self):
        return len(self.ims)

    def get_input_mask(self, imgpath, view): # index: denotes camera idx
        

        maskpath = imgpath.replace('raw','annotations/matting/').replace('.jpg', '.png')
       
   

        mask = (imageio.imread(maskpath) > 127).astype(np.uint8)[:, self.clip:-self.clip]

        return mask

    def __getitem__(self, idx):
        # sample a frame for training
        
        tar_img_path = self.ims[idx]
        human = tar_img_path.split('_e_')[0].split('/')[-1]

        input_view = copy.deepcopy(self.test_input_view)
        all_input_view = copy.deepcopy(self.test_views)


        input_view = copy.deepcopy(self.test_input_view)

        # select a target view
        
        
        if self.setting2:
            input_view = self.views[idx]
            tar_view_ind = input_view[0]
        else:
            tar_pool = list(set(all_input_view) - set(input_view))
            random.shuffle(tar_pool)
            tar_view_ind = tar_pool[0]
            input_view = [tar_view_ind] + input_view

        input_imgs, input_msks, input_cameras = [], [], []
        # input_msks_dial = []
        erode_border = 5
        
        index_num = idx
        for index_view,view in enumerate(input_view):
            input_img_path = tar_img_path % view
            input_msk = self.get_input_mask(input_img_path, view)
            # load data
            
            in_K, in_D = np.array(self.cams[human]['%02d'%view]['K']).astype(np.float32), np.array(self.cams[human]['%02d'%view]['D']).astype(np.float32)
            c2w = np.array(self.cams[human]['%02d'%view]['RT']).astype(np.float32)
            input_img = imageio.imread(input_img_path).astype(np.float32)[:, self.clip:-self.clip] / 255.
            #input_img, input_msk = cv2.undistort(input_img, in_K, in_D), cv2.undistort(input_msk, in_K, in_D)


            original_H,original_W =input_img.shape[0],input_img.shape[1]
            # resize images
            H, W = int(input_img.shape[0] * self.ratio), int(input_img.shape[1] * self.ratio)
            input_img, input_msk = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)

            
            # apply foreground mask
            input_img[input_msk == 0] = 1
            
            
            in_K[0,2] = in_K[0,2] - float(self.clip)
            in_K[:2] = in_K[:2] * self.ratio
            K = np.eye(4)
            K[:3, :3] = in_K.copy()
            camera = np.concatenate((list(input_img.shape[:2]), K.flatten(), c2w.flatten())).astype(np.float32)

            
            # append data
            input_imgs.append(input_img)
            input_msks.append(input_msk)
            input_cameras.append(camera)
    
    
        src_rgbs = np.stack(input_imgs[1:])
        src_cameras = np.stack(input_cameras[1:])
        rgb = input_imgs[0]
        rgb_file = tar_img_path
        camera = input_cameras[0]

        
        depth_range = torch.tensor([0.5, 2.5])
        
        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                'tar_view':tar_view_ind,

                }

