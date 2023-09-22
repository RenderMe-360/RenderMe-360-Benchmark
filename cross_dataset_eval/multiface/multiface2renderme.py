import numpy as np
import struct
import cv2
import sys
import os
import glob
import multiprocessing
from tqdm import tqdm
import pickle


def load_KRT(fname):
    with open(fname, 'r') as fp:
        text = fp.readlines()
    Ks = []; w2cs= []; dists = []; camIDs = []
    K  = []; w2c = []; dist  = []; cam = 0
    dtype = np.float32
    for i in range(len(text)):
        try:
            l = [float(_) for _ in text[i].strip().split(' ') if len(_) > 0]
        except	Exception as e:
            continue
        if len(l) <= 0:
            continue
        elif len(l) == 1:
            cam = int(l[0])
        elif len(l) == 3:
            K.append(l)
        elif len(l) == 4:
            w2c.append(l)
        elif len(l) == 5:
            dist = l
        if len(w2c) >= 3 and len(K) >= 3 and cam > 0:
            w2c.append([0,0,0,1])
            Ks.append(np.array(K, dtype))
            w2cs.append(np.array(w2c, dtype))
            dists.append(np.array(dist, dtype))
            camIDs.append(cam)
            K = []; w2c = []; dist = []; cam = 0
    return	Ks, w2cs, dists, camIDs
def gamma_correct(bgr):
    g, black, scale = 2.0, 3./255., [1.4,1.1,1.6]
    bgr = bgr.astype(np.float32) / 255.
    scale = np.reshape(scale[::-1], [1,1,3]).astype(bgr.dtype) # rgb2bgr
    bgr = bgr * scale / 1.1
    bgr = np.clip(((1/(1-black))*0.95*np.clip(bgr-black,0,2))**(1./g) - 15./225, 0, 2)
    bgr = np.clip(np.round(bgr * 255), 0, 255).astype(np.uint8)
    return	bgr
def load_bin(fname, dtype = np.float32):
	with open(fname, 'rb') as fid:
		text = fid.read()
	l = len(text) // 4 // 3
	v = []
	for i in range(l):
		v.append(struct.unpack('fff', text[i*12:i*12+12]))
	v = np.array(v, dtype)
	return	v
def load_obj(fname, vtype = np.float32, ftype = np.int64):
	v = np.zeros([0,3], vtype)
	fv= np.zeros([0,3], ftype)
	if fname[-4:].lower() != '.obj' or not os.path.exists(fname):
		return	v, fv
	with open(fname, 'r') as fid:
		text = fid.readlines()
		v = []; fv= []
	for line in text:
		line = [s for s in line.strip().split(' ') if len(s) > 0]
		if len(line) >= 4 and line[0] == 'v':
			v += [[float(f) for f in line[1:]]]
		elif len(line) >= 4 and line[0] == 'f':
			f = [[],[],[]]
			for i in range(1, len(line)):
				l = line[i].split('/')
				for j in range(3):
					if j < len(l) and len(l[j]) > 0:
						f[j] += [int(l[j]) - 1]
			fv += [[f[0][0], f[0][i-1], f[0][i]] for i in range(2,len(f[0]))]
	v = np.array(v, vtype)
	fv= np.array(fv,ftype)
	return	v, fv
def draw(Img, points, r = 1, color = (0,255,0)):
	points = np.array(points)
	dim = points.shape[-1] if len(points.shape) >= 2 else 2
	if dim == 1:
		points = points.reshape(-1,2)
		dim = 2
	Img = Img.copy()
	for p in points.reshape(-1,dim):
		if np.isnan(p).sum() > 0: continue
		x = int(min(max(p[0],-abs(r)),Img.shape[1]+abs(r)))
		y = int(min(max(p[1],-abs(r)),Img.shape[0]+abs(r)))
		cv2.circle(Img, (x,y), abs(r), color, -1 if r > 0 else 1)
	return	Img
def FLAME(model, **kwargs):
	v = model['v_template'].reshape(-1,3)
	if 'shape' in kwargs.keys():
		s = kwargs['shape'].reshape(-1)
		dim = min(len(s), model['shapedirs'].shape[-1])
		if dim > 0:
			v_shaped = v + model['shapedirs'].reshape(len(v)*3,-1) \
				[:,:dim].dot(s[:dim]).reshape(v.shape)
		else:
			v_shaped = v
	else:
		v_shaped = v
	if 'exp' in kwargs.keys():
		e = kwargs['exp'].reshape(-1)
		w = model['shapedirs'][...,-100:]
		dim = min(len(e), w.shape[-1])
		if dim > 0:
			v_shaped += w.reshape(len(v)*3,-1) \
				[:,:dim].dot(e[:dim]).reshape(v.shape)
	J = model['J_regressor'].dot(v_shaped)
	pose = np.zeros([model['kintree_table'].shape[1], 3])
	for k in [key for key in kwargs.keys()]:
		if 'pose' in k:
			p = kwargs[k].reshape(-1)
			p = cv2.Rodrigues(p[:9].reshape(3,3))[0].reshape(-1) \
				if len(p) >= 9 else p[:3]
			kwargs[k] = p
		if k == 'global_pose':
			pose[0,:] = p
		if k == 'neck_pose':
			pose[1,:] = p
		elif k == 'jaw_pose':
			pose[2,:] = p
		elif k == 'left_eye_pose':
			pose[3,:] = p
		elif k == 'right_eye_pose':
			pose[4,:] = p; has_pose = True
	R = np.array([cv2.Rodrigues(_)[0] for _ in pose])
	p = np.concatenate([(R_-np.identity(3)).reshape(-1) for R_ in R[1:]])
	v_posed = v_shaped + model['posedirs'].dot(p).reshape(v.shape)
	T = [[] for _ in range(len(J))]; v = 0
	for p, c in model['kintree_table'].T:
		if p < 0 or p >= len(T):
			T[c] = np.concatenate([R[c], J[c:c+1,:].T],1)
		else:
			T[c] = np.concatenate([ \
				T[p][:3,:3].dot(R[c]), \
				T[p][:3,3:4] + T[p][:3,:3].dot((J[c:c+1,:]-J[p:p+1,:]).T)],1)
		v = v + model['weights'][:,c:c+1] * \
			((v_posed - J[c:c+1,:]).dot(T[c][:3,:3].T) + T[c][:3,3:4].T)
	if 'cam' in kwargs.keys():
		v = v + kwargs['cam'].reshape(1,-1)[:,1:]
	return	v

def func(path_img, write_krt=False):
    subject = path_img.split('/')[-5]
    exp = path_img.split('/')[-3]
    cam = path_img.split('/')[-2]
    name = path_img.split('/')[-1]
    fr = name.split('.')[0]
    data_dir = '/'.join(path_img.split('/')[:-4])
    out_dir = os.path.join(dir_out, exp, cam)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    matting_out_dir = os.path.join(dir_matting_out, exp, cam, 'pha')
    if not os.path.exists(matting_out_dir):
        os.makedirs(matting_out_dir, exist_ok=True)
    # world to camera
    K, w2c, dist, IDs = load_KRT(os.path.join(data_dir,'KRT'))
    i = IDs.index(int(cam))
    K, w2c, dist = K[i], np.concatenate([w2c[i][:3,:],[[0,0,0,1]]], 0), dist[i]
    # multiface-average to world
    f2w = np.loadtxt(os.path.join(data_dir,'tracked_mesh', exp, fr+'_transform.txt'))
    f2w = np.concatenate([f2w,[[0,0,0,1]]], 0)
    # renderme-flame to multiface
    r2f = np.array([ \
        [965.5675, -17.3101,  25.3721, -1.4875], \
        [ 20.1298, 959.3857,-111.5253, 19.3127], \
        [-23.1985, 111.9976, 959.2614, -6.3881], \
        [0,0,0,1]], f2w.dtype)

    renderme_w2c = np.array([ \
        np.linalg.inv(np.concatenate([ \
        renderme_cam[_]['RT'][:3,:], [[0,0,0,1]]], 0)) \
        for _ in renderme_views])
    r2c = w2c.dot(f2w).dot(r2f).dot(np.linalg.inv(r2w)) # renderme world coor. to multiface camera coor.
    scale = np.linalg.det(r2c[:3,:3])**(1./3)
    r2c[:3,:] /= scale
    metric  = [r2c[2,:3].dot(w2c[2,:3]) for w2c in renderme_w2c] # find the nearst pose
    nearest = np.argmax(metric)
    renderme_K, renderme_w2c= renderme_cam[renderme_views[nearest]]['K'], \
                renderme_w2c[nearest]
    Rz = renderme_w2c.dot(np.linalg.inv(r2c)) # multiface camera coor. to renderme camera coor.
    roll = np.arctan2(Rz[1,0]-Rz[0,1], Rz[0,0]+Rz[1,1])
    zmean = r2c[2,3] * 9.0
    tx = Rz[0,3] / (r2c[2,3] + zmean)
    ty = Rz[1,3] / (r2c[2,3] + zmean)
    Rz = np.array([	[np.cos(roll),-np.sin(roll),tx], \
            [np.sin(roll), np.cos(roll),ty], \
            [0,0,1]], Rz.dtype) # affine transform
    A = renderme_K.dot(Rz[:3,:3]).dot(np.linalg.inv(K)) # multiface uv to renderme uv
    K_new  = renderme_K.copy()
    K_new[:2,2] += K_new[:2,:2].dot([tx,ty])

    bgr = cv2.imread(os.path.join(data_dir,'images',exp,'%s'%cam, fr+'.png'))
    bgr = gamma_correct(bgr)
    bgr = cv2.warpAffine(bgr, A[:2,:], (2448,2048))
    cv2.imwrite(os.path.join(out_dir, name), bgr)
    Rz = np.array([	[Rz[0,0],Rz[0,1],0,0], \
                    [Rz[1,0],Rz[1,1],0,0], \
                    [0,0,1,0],[0,0,0,1]])
    w2c_new= Rz.dot(r2c)
    # matting. NOTE: need to run the background matting from original image in advance
    mask = cv2.imread(os.path.join(dir_matting, exp,'%s'%cam, fr+'.png'))
    mask = cv2.warpAffine(mask, A[:2,:], (2448,2048))
    cv2.imwrite(os.path.join(matting_out_dir, name), mask)

    if write_krt:
        fid.writelines(f"{cam}\n")
        fid.writelines(f"{K_new[0,0]:9.4f} {K_new[0,1]:9.4f} {K_new[0,2]:9.4f}\n")
        fid.writelines(f"{K_new[1,0]:9.4f} {K_new[1,1]:9.4f} {K_new[1,2]:9.4f}\n")
        fid.writelines(f"{K_new[2,0]:9.4f} {K_new[2,1]:9.4f} {K_new[2,2]:9.4f}\n")
        fid.writelines(f"0.0 0.0 0.0 0.0 0.0\n")
        fid.writelines(f"{w2c_new[0,0]:11.8f} {w2c_new[0,1]:11.8f} {w2c_new[0,2]:11.8f} {w2c_new[0,3]:11.8f}\n")
        fid.writelines(f"{w2c_new[1,0]:11.8f} {w2c_new[1,1]:11.8f} {w2c_new[1,2]:11.8f} {w2c_new[1,3]:11.8f}\n")
        fid.writelines(f"{w2c_new[2,0]:11.8f} {w2c_new[2,1]:11.8f} {w2c_new[2,2]:11.8f} {w2c_new[2,3]:11.8f}\n")
        fid.writelines("\n")


base_dir = "./multiface/" # TODO: input your path to dataset
subject = sys.argv[1]
print(subject)
dir_img = os.path.join(base_dir, subject, 'images')
dir_out = os.path.join(base_dir, subject, 'images_gamma_aligned')
dir_matting = os.path.join(base_dir, subject, 'matting')
dir_matting_out = os.path.join(base_dir, subject, 'matting_aligned')
path_imgs = sorted(glob.glob(f"{dir_img}/*/*/*.png"))

# renderme-flame to renderme-world
flame_path = 'flame_example.npy'
flame = np.load(flame_path, allow_pickle = True)
flame = flame.reshape(-1)[0]
with open('generic_model.pkl', 'rb') as fp:
    model = pickle.load(fp, encoding = 'latin1')
v_shaped = FLAME(model, shape = flame.get('shape',[]), exp = flame.get('exp',[]))
J = model['J_regressor'].dot(v_shaped)
R = cv2.Rodrigues(flame['global_pose'])[0]
r2w = np.concatenate([np.concatenate([R, \
    (J[0,:]-R.dot(J[0,:]) + flame['cam'].reshape(-1)[1:]).reshape(-1,1)],1),\
    [[0,0,0,1]]], 0)
renderme_annot_dir='camera_para_example.npy'
renderme_cam = np.load(renderme_annot_dir, allow_pickle = True)
renderme_cam = renderme_cam.reshape(-1)[0]['cams']
renderme_views=sorted([_ for _ in renderme_cam.keys()])

# multi process
max_processes = 5
pool = multiprocessing.Pool(processes=max_processes)

pbar = tqdm(total=len(path_imgs))
pbar.set_description(subject)
update = lambda *args: pbar.update()

fid = None
processes = set()
for path_img in tqdm(path_imgs):
    # func(path_img)
    pool.apply_async(func, (path_img,), callback=update)
pool.close()
pool.join()

dict_cam = {}
for path_img in tqdm(path_imgs):
    cam = path_img.split('/')[-2]
    if cam not in dict_cam.keys():
        dict_cam[cam] = path_img
path_krt_new = os.path.join(base_dir, subject, 'KRT_align.txt')
fid = open(path_krt_new, 'w')
for cam in dict_cam.keys():
    func(dict_cam[cam], True)
print("done")
