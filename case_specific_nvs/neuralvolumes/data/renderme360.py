# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from matplotlib.pyplot import axis
import numpy as np
import os, sys, random
from PIL import Image
import imageio, cv2
from .utils import load_ply, load_obj

import torch.utils.data


def read_image(path):
    if path.endswith('.png'):
        mask = imageio.imread(path)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        return mask
    else:
        image = imageio.imread(path)
        return image

def read_annots(path):
    annots = np.load(path, allow_pickle=True)
    return annots.item()

def listdir(path):
    image_paths = os.listdir(path)
    return image_paths

def read_lmk3d(path):
    lmk3d = np.load(path, allow_pickle=True)
    return lmk3d

def gen_cam_views(transl, z_pitch, viewnum):
    def viewmatrix(z, up, translation):
        vec3 = z / np.linalg.norm(z)
        up = up / np.linalg.norm(up)
        vec1 = np.cross(up, vec3)
        vec2 = np.cross(vec3, vec1)
        view = np.stack([vec1, vec2, vec3, translation], axis=1)
        view = np.concatenate([view, np.array([[0,0,0,1]])], axis=0)
        return view

    cam_poses = []
    for i, theta in enumerate(np.linspace(-np.pi/2, 1.5*np.pi, viewnum+1)[:-1]):
        theta = -theta
        dist = 1.5
        
        z = np.array([np.cos(theta), 0, np.sin(theta)])
        t = -z * dist + transl

        z = z * np.sqrt(1-z_pitch*z_pitch)
        z[1] = z_pitch
        z = z * dist
        up = np.array([0,1,0])
        view = viewmatrix(z, up, t)
        cam_poses.append(view)
    return cam_poses

def image_cropping(mask):
    a = np.where(mask != 0)
    h, w = list(mask.shape[:2])

    top, left, bottom, right = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    bbox_h, bbox_w = bottom - top, right - left
    if bbox_h < (0.5 * h) or bbox_w < (0.5 * w):
        return 0, 0, min(h, w), min(h, w)
    # padd bbox
    bottom = min(int(bbox_h*0.1+bottom), h)
    top = max(int(top-bbox_h*0.1), 0)
    right = min(int(bbox_w*0.1+right), w)
    left = max(int(left-bbox_h*0.1), 0)
    bbox_h, bbox_w = bottom - top, right - left

    if bbox_h < bbox_w:
        w_c = (left+right) / 2
        size = bbox_h
        if w_c - size / 2 < 0:
            left = 0
            right = size
        elif w_c + size / 2 >= w:
            left = w - size
            right = w
        else:
            left = int(w_c - size / 2)
            right = left + size
    else:
        h_c = (top+bottom) / 2
        size = bbox_w
        if h_c - size / 2 < 0:
            top = 0
            bottom = size
        elif h_c + size / 2 >= h:
            top = h - size
            bottom = h
        else:
            top = int(h_c - size / 2)
            bottom = top + size
    
    return top, left, bottom, right


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datadir, annotdir, subject, keyfilter, ws_factor=1.0,
            fixedcammean=0., fixedcamstd=1., imagemean=0., 
            imagestd=1., subsampletype=None, subsamplesize=0,
            loadSize=512, move_cam=0, mode='sequence', cam=None, train_views=list(range(60)),test_views=[]):
        self.move_cam = move_cam
        self.subject = subject
        self.datadir = datadir
        self.annotdir = annotdir
        self.date = self.annotdir.split('/')[-2]
        self.annots = read_annots(annotdir)['cams']
        self.all_cameras = sorted([int(k) for k in self.annots.keys()])
        self.cameras = list(range(60))
        self.cameras = [str(cam) for cam in self.cameras if cam in self.all_cameras]
        self.fixedcameras = [1, 13, 25, 37]
        self.fixedcameras = [cam for cam in self.fixedcameras if cam in self.all_cameras]
        self.loadSize = loadSize
        self.train_views = train_views
        self.test_views = test_views if cam is None else [cam]
        
        assert mode in ['sequence', 'exp', 'speech', 'hair']
        self.mode = mode
        self.ninput = len(self.fixedcameras)
        self.is_train = subsampletype != None

        if self.mode == 'exp':
            self.framelist = []
            if self.is_train:
                for i in range(10, 12):
                    frame_dir = os.path.join(self.datadir, subject, f'{subject}_e_{i}/images')
                    self.framelist += [os.path.join(frame_dir, '{}', _f) for _f in listdir(os.path.join(frame_dir, '00'))[::2]]
                self.framecamlist = [(x[:-4], cam)
                        for x in self.framelist
                        for cam in self.train_views]
            else:
                for i in range(10, 12):
                    frame_dir = os.path.join(self.datadir, subject, f'{subject}_e_{i}/images')
                    # self.framelist += [os.path.join(frame_dir, '{}', _f) for _f in listdir(os.path.join(frame_dir, '00'))]
                    mask_dir = os.path.join(self.datadir, subject, f'{subject}_e_{i}/matting') # TODO:只会取有mask的一帧
                    # print(f'mask_dir test: {mask_dir}')
                    self.framelist += [os.path.join(frame_dir, '{}', _f) for _f in listdir(os.path.join(mask_dir, '00'))]
                    self.framelist.sort();self.framelist = self.framelist[-30:-29]
                if self.move_cam == 0:
                    self.framecamlist = [(x[:-4], cam)
                        for x in self.framelist
                        for cam in self.test_views]
                    self.framelist = self.framelist
                else:
                    self.framecamlist = [(x[:-4], 25) for x in self.framelist]
                    self.render_path = self.get_render_path()
                    self.framelist = self.framelist
            # print(f"self.framecamlist: {self.framecamlist }")
        elif self.mode == 'hair':
            self.framelist = []
            if self.is_train:
                frame_dir = os.path.join(self.datadir, subject, f'{subject}_e_{i}/images')
                self.framelist += [os.path.join(frame_dir, '{}', _f) for _f in listdir(os.path.join(frame_dir, '00'))]
                self.framelist = self.framelist[:90] + self.framelist[120:] # NOTE: split train and test frame, novel time stamp
            
                # print(f'frame_dir: {frame_dir}')
                # mask_dir = os.path.join(self.datadir, 'MASK', self.date, f'{subject}_e_{i}')
                # self.framelist += [os.path.join(mask_dir, '{}', _f) for _f in listdir(os.path.join(mask_dir, '00'))[::2]]
                self.framecamlist = [(x[:-4], cam)
                        for x in self.framelist
                        for cam in self.train_views]
            else:
                frame_dir = os.path.join(self.datadir, subject, f'{subject}_e_{i}/images')
                self.framelist += [os.path.join(frame_dir, '{}', _f) for _f in listdir(os.path.join(frame_dir, '00'))]
                # self.framelist = self.framelist[90:120] # NOTE: split train and test frame, novel time stamp
                # mask_dir = os.path.join(self.datadir, 'MASK', self.date, f'{subject}_e_{i}') # TODO:只会取有mask的一帧
                # self.framelist += [os.path.join(mask_dir, '{}', _f) for _f in listdir(os.path.join(mask_dir, '00'))]
                if self.move_cam == 0:
                    self.framecamlist = [(x[:-4], cam)
                        for x in self.framelist
                        for cam in self.test_views]
                    self.framelist = self.framelist
                else:
                    self.framecamlist = [(x[:-4], 25) for x in self.framelist]
                    self.render_path = self.get_render_path()
                    self.framelist = self.framelist

        else:
            raise NotImplementedError
        # print("framelist length: ", len(self.framelist))
        self.keyfilter = keyfilter
        self.fixedcammean = fixedcammean
        self.fixedcamstd = fixedcamstd
        self.imagemean = imagemean
        self.imagestd = imagestd
        self.subsampletype = subsampletype
        self.subsamplesize = subsamplesize
        self.cropping = 'cropping' in self.keyfilter
        # NOTE: try to find the best world scale
        self.worldscale = 2.0 / ws_factor
        print('worldscale: ', self.worldscale)
        
        # immitate dryice dataloader which is basically a cv to gl transformation
        self.transf = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        
        # NOTE: load background images for each camera
        if "bg" in self.keyfilter:
            self.bg = {}
            for i, cam in enumerate(self.cameras):
                try:
                    # print(self.annotdir)
                    date = self.annotdir.split('/')[-2]
                    image = 255 * np.ones((512, 512, 3))
                    image = np.asarray(image).transpose((2, 0, 1)).astype(np.float32)
                    self.bg[cam] = image
                except Exception as e:
                    print('Error: no bg images!\n##########################')
                    print(e)
                    # exit(-1)
                    pass

    def get_allcameras(self):
        return self.all_cameras

    def known_background(self):
        return "bg" in self.keyfilter

    def get_background(self, bg):
        if "bg" in self.keyfilter:
            for i, cam in enumerate(self.cameras):
                if cam in self.bg:
                    bg[cam].data[:] = torch.from_numpy(self.bg[cam]).to("cuda")

    def get_krt(self):
        return {str(k): {
                #"pos": self.campos[k],
                #"rot": self.camrot[k],
                #"focal": self.focal[k],
                #"princpt": self.princpt[k],
                "size": np.array([512, 512])}
                for k in self.cameras}

    def circle_path_from_flame(self, num_views):
        height, pitch = [], []
        for view in range(1,48,3):
            view = '%02d' % view
            if view in self.annots.keys():
                height.append(self.annots[view]['RT'][1, 3])
                z_rodrigous = self.annots[view]['RT'][:3,:3]@np.array([[0],[0],[1]])
                pitch.append(z_rodrigous[1,0])
        transl = np.array([0, np.mean(np.array(height)), 0])
        z_pitch = np.mean(np.array(pitch))

        render_poses = gen_cam_views(transl, z_pitch, num_views)
        render_poses = [pose for pose in render_poses]

        return np.array(render_poses).astype(np.float32)

    def load_image(self, frame, cam):
        cam = int(cam)
        # imagedir = os.path.join(self.datadir, 'IMAGES', self.subject, f'{'%02d'%cam}')
        # imagepaths = sorted([os.path.join(imagedir, dir_) for dir_ in listdir(imagedir) if frame in dir_])
        imagepath = frame.format('%02d'%cam) + '.jpg'
        # print(f'imagepath: {imagepath}')
        image = read_image(imagepath)
        K = np.array(self.annots['%02d'%cam]['K']).astype(np.float32).reshape(3, 3)
        D = np.array(self.annots['%02d'%cam]['D']).astype(np.float32).reshape(5,)
        c2w = np.array(self.annots['%02d'%cam]['RT']).astype(np.float32).reshape(4, 4)
        # image = cv2.undistort(image, K, D)

        # maskdir = os.path.join(self.datadir, 'MATTING_NEW', 'R101', self.subject, '%02d'%cam, 'pha')
        maskpath = imagepath.replace('images', 'matting')
        maskpath = maskpath.replace('.jpg', '.png')
        maskpath = '/'.join(maskpath.split('/')[:-1] + [maskpath.split('/')[-1]])
        # print(f"maskpath: {maskpath}")
        mask = read_image(maskpath)
        mask = mask.astype(np.float32) / 255.
        if len(mask.shape) == 3:
            mask = mask[...,0]
        image = image * mask[...,None] + (1 - mask[...,None]) * 255
        # st()
        top, left, bottom, right = 0, 200, 2048, 2248 # image_cropping(mask)
        assert (bottom - top) == (right - left)
        image = image[top:bottom, left:right]
        image = cv2.resize(image.copy(), (self.loadSize, self.loadSize))

        # new_h, new_w = image.shape[:2]
        image = image.transpose((2,0,1)).astype(np.float32)
        # c2w = np.concatenate([RT, np.array([0, 0, 0, 1]).reshape(1, 4).astype(np.float32)], axis=0)
        K[0,2] -= left
        K[1,2] -= top
        K[0,:] *= self.loadSize / float(right - left)
        K[1,:] *= self.loadSize / float(bottom - top)
        return image, K, c2w

    def get_render_path(self):
        height, pitch = [], []
        for view in range(1,48,3):
            view = '%02d' % view
            if view in self.annots.keys():
                height.append(self.annots[view]['RT'][1, 3])
                z_rodrigous = self.annots[view]['RT'][:3,:3]@np.array([[0],[0],[1]])
                pitch.append(z_rodrigous[1,0])
        transl = np.array([0, np.mean(np.array(height)), 0])
        z_pitch = np.mean(np.array(pitch))

        render_poses = gen_cam_views(transl, z_pitch, len(self.framecamlist))

        return render_poses

    def __len__(self):
        return len(self.framecamlist)

    def __getitem__(self, idx):
        frame, cam = self.framecamlist[idx]

        result = {}
        validinput = True

        # fixed camera images
        if "fixedcamimage" in self.keyfilter:
            
            fixedcamimage = []
            fixed_Ks = []
            for i in range(self.ninput):
                fixedimg, fixedK, fixedRt = self.load_image(frame, self.fixedcameras[i])
                # cv2.imwrite(f'tmp/fixed_{i}_{idx}.jpg', fixedimg.transpose((1, 2, 0))[..., ::-1])
                fixedcamimage.append(fixedimg)
                fixed_Ks.append(fixedK.reshape(1, 3, 3))
            fixed_Ks = np.concatenate(fixed_Ks, axis=0)
            fixedcamimage = np.concatenate(fixedcamimage, axis=0)
            fixedcamimage[:] -= self.imagemean
            fixedcamimage[:] /= self.imagestd
            result["fixedcamimage"] = fixedcamimage

        result["validinput"] = np.float32(1.0 if validinput else 0.0)

        # image data
        if cam is not None:
            if "camera" in self.keyfilter or "image" in self.keyfilter:
                image, K, Rt = self.load_image(frame, cam)
                # cv2.imwrite(f'tmp/src_{idx}.jpg', image.transpose((1, 2, 0))[..., ::-1])
                if self.move_cam > 0:
                    Rt = self.render_path[idx].copy().astype(np.float32)
                    # Rt = Rt @ gl2cv
                # camera data
                # w2c @ trasf * worldscale
                result["camrot"] = (np.linalg.inv(Rt) @ self.transf)[:3,:3] * self.worldscale
                pts3d_path = self.framecamlist[0][0].replace('images', 'lmk3d')
                # print(f'pts3d_path: {pts3d_path}')
                pts3d_path = '/'.join(pts3d_path.split('/')[:-3] + [pts3d_path.split('/')[-3],pts3d_path.split('/')[-1]]) + '.npy'
                print(f'pts3d_path: {pts3d_path}')
                pts3d = read_lmk3d(pts3d_path).astype(np.float32)
                center = (pts3d.min(0) + pts3d.max(0)) / 2
                # center = np.array([0, -0.46, -0.07]).astype(np.float32)
                result["campos"] = np.dot(self.transf[:3, :3].T, Rt[:3, 3] - center) * self.worldscale
                result["focal"] = np.diag(K[:2, :2])
                result["princpt"] = K[:2, 2]
                result["camindex"] = self.all_cameras.index(int(cam))
                # result["frameid"] = frame

                result["image"] = image
                result["imagevalid"] = np.float32(1.0)
                

            if "pixelcoords" in self.keyfilter:
                if self.subsampletype == "patch":
                    indx = np.random.randint(0, self.loadSize - self.subsamplesize + 1)
                    indy = np.random.randint(0, self.loadSize - self.subsamplesize + 1)

                    px, py = np.meshgrid(
                            np.arange(indx, indx + self.subsamplesize).astype(np.float32),
                            np.arange(indy, indy + self.subsamplesize).astype(np.float32))
                elif self.subsampletype == "random":
                    px = np.random.randint(0, self.loadSize, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                    py = np.random.randint(0, self.loadSize, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                elif self.subsampletype == "random2":
                    px = np.random.uniform(0, self.loadSize - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                    py = np.random.uniform(0, self.loadSize - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                else:
                    px, py = np.meshgrid(np.arange(self.loadSize).astype(np.float32), np.arange(self.loadSize).astype(np.float32))

                result["pixelcoords"] = np.stack((px, py), axis=-1)

        return result
