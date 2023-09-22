from tqdm import tqdm
import numpy as np
import pickle
import struct
import json
import cv2
import sys
import os


import torch
import _utils

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
		w2c =np.concatenate([w2c[:3,:],[[0,0,0,1]]],0)
		h = int(data['%d_height' % ID])
		w = int(data['%d_width'  % ID])
		valid = data.get('%d_valid' % ID, False)
		if valid:
			Ks.append(K)
			w2cs.append(w2c)
			dists.append(dist)
			camIDs.append(ID)
	return	Ks, w2cs, dists, camIDs
def load_bin(fname, dtype = np.float32):
	with open(fname, 'rb') as fid:
		text = fid.read()
	l = len(text) // 4 // 3
	v = []
	for i in range(l):
		v.append(struct.unpack('fff', text[i*12:i*12+12]))
	v = np.array(v, dtype)
	return	v
def load_ply(fname, vtype = np.float32, ftype = np.int64):
	v = np.zeros([0,3], vtype)
	fv= np.zeros([0,3], ftype)
	if fname[-4:].lower() != '.ply' or not os.path.exists(fname):
		return	v, fv
	def readl(f):
		l = str(f.readline())
		if sys.version_info[0] >= 3:
			l = l[2:-1].replace('\\n','\n') \
				.replace('\\r','\r') \
				.replace('\\t','\t')
		return	l.strip() if len(l) > 0 else None
	def type2struct(s):
		s = str(s).lower()
		if s == 'float':
			return	'f'
		elif s == 'double':
			return	'd'
		elif s == 'char':
			return	'b'
		elif s == 'uchar':
			return	'B'
		elif s == 'short':
			return	'h'
		elif s == 'ushort':
			return	'H'
		elif s == 'int':
			return	'i'
		elif s == 'uint':
			return	'I'
		else:
			return	''
	sizeof = {'f': 4, 'd': 8, 'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4}
	with open(fname, 'rb') as fid:
		head = readl(fid)
		if str(head).lower() != 'ply':
			return	v, fv
		l = readl(fid); vnum = 0; fnum = 0; vtype = ''; ftype = ''; fmt = '?'
		while l is not None:
			l = [_.lower() for _ in l.split(' ') if len(_) > 0]
			if len(l) <= 1:
				pass
			elif l[0] == 'format' and len(l) >= 2:
				if 'bin' in l[1]:
					fmt = '<' if not 'big' in l[1] else '>'
			elif l[0] == 'element':
				if l[1] == 'vertex' and len(l) >= 3:
					vnum = int(l[2]); vtype = fmt
				elif l[1] == 'face' and len(l) >= 3:
					fnum = int(l[2])
			elif l[0] == 'property':
				if l[1] == 'list' and len(l) >= 5:
					ftype = fmt+type2struct(l[2])+type2struct(l[3])
				else:
					vtype+= type2struct(l[1])
			l = readl(fid)
			if 'end_header' in l: break
		v = np.zeros([vnum, len(vtype)-1], v.dtype); tri = []
		if vtype[0] == '?':
			for i in tqdm(range(vnum+fnum), desc='loading %s'%os.path.basename(fname)):
				if i < vnum:
					l = [_ for _ in l.split(' ') if len(_) > 0][:len(vtype)-1]
					v[i,:len(l)] = [float(l[_]) if vtype[_+1] in 'fd' else \
						int(l[_]) for _ in range(len(l))]
				else:
					l = [int(_) for _ in l.split(' ') if len(_) > 0]
					if len(l) <= 0: continue
					tri += [(l[1],l[j-1],l[j]) \
						for j in range(3,min(len(l),l[0]+1))]
		else:
			n = sum([sizeof[_] for _ in vtype[1:]])
			for i in tqdm(range(vnum+fnum), desc='loading %s'%os.path.basename(fname)):
				if i < vnum:
					l = fid.read(n)
					if len(l) < n: break
					v[i,:] = struct.unpack(vtype, l)
				else:
					n = sizeof[ftype[1]]
					l = fid.read(n)
					if len(l) < n: break
					n = struct.unpack(ftype[:2], l)[0]
					l = fid.read(sizeof[ftype[2]]*n)
					if len(l) < sizeof[ftype[2]]*n: break
					l = struct.unpack(ftype[0]+ftype[2]*n, l)
					tri += [(l[0],l[j-1],l[j]) for j in range(2,len(l))]
		fv = np.array(tri, fv.dtype).reshape(-1,3)
	return	v, fv
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
def load_sRt(file_name, ID = 0, exp = '1_neutral'):
	with open(file_name, 'r') as fid:
		data = json.load(fid)
	if isinstance(exp, str):
		exp = int(exp.split('_')[0])
	sRt = data['%d' % ID]['%d' % exp]
	s = sRt[0]
	Rt = np.array(sRt[1], np.float64)
	align2world = Rt
	align2world = np.linalg.inv(np.concatenate([align2world,[[0,0,0,1]]],0))
	align2world[:3,:] /= s
	return	align2world
def zbuffer(v, tri, zbuf, persp = True, double_face = True):
	v = np.array(v)
	eps = np.finfo(v.dtype).eps if np.array(1.1,v.dtype)>1 else 1
	h, w = zbuf.shape[:2]
	zbuf = zbuf.reshape(h,w,-1)[...,0]
	v_proj = v[:,:2] / np.maximum(v[:,2:3],eps) if persp else v[:,:2]
	va = v_proj[tri[:,0],:2]
	vb = v_proj[tri[:,1],:2]
	vc = v_proj[tri[:,2],:2]
	front = np.cross(vc-va, vb-va) if not double_face else \
		np.ones_like(va[:,0])
	umin = np.maximum(np.ceil (np.vstack((va[:,0],vb[:,0],vc[:,0])).min(0)), 0)
	umax = np.minimum(np.floor(np.vstack((va[:,0],vb[:,0],vc[:,0])).max(0)),w-1)
	vmin = np.maximum(np.ceil (np.vstack((va[:,1],vb[:,1],vc[:,1])).min(0)), 0)
	vmax = np.minimum(np.floor(np.vstack((va[:,1],vb[:,1],vc[:,1])).max(0)),h-1)
	umin = umin.astype(np.int32)
	umax = umax.astype(np.int32)
	vmin = vmin.astype(np.int32)
	vmax = vmax.astype(np.int32)
	front = np.where(np.logical_and(np.logical_and( \
		umin <= umax, vmin <= vmax), front > 0))[0]
	for t in front:
		A = np.concatenate((vb[t:t+1]-va[t:t+1], vc[t:t+1]-va[t:t+1]),0)
		x, y = np.meshgrid(	range(umin[t],umax[t]+1), \
					range(vmin[t],vmax[t]+1))
		u = np.vstack((x.reshape(-1),y.reshape(-1))).T
		coeff = (u.astype(v.dtype) - va[t:t+1,:]).dot(np.linalg.pinv(A))
		coeff = np.concatenate((1-coeff.sum(1).reshape(-1,1),coeff),1)
		if persp:
			z = coeff.dot(v[tri[t], 2])
		else:
			z = 1 / np.maximum((coeff/v[tri[t],2:3].T).sum(1), eps)
		for i, (x, y) in enumerate(u):
			if  coeff[i,0] >= -eps \
			and coeff[i,1] >= -eps \
			and coeff[i,2] >= -eps \
			and zbuf[y,x] > z[i]:
				zbuf[y,x] = z[i]
	return	zbuf

def zbuffer_fast(v, tri, zbuf):
	v = torch.from_numpy(np.array(v).astype(np.float64))
	tri=torch.from_numpy(tri.astype(np.int64))
	zbuf = torch.from_numpy(zbuf.astype(np.float64))
	_, zbuf = _utils.vis(v, v, tri, -torch.ones([len(v),2], dtype = torch.int64), zbuf)
	zbuf = zbuf.detach().cpu().numpy()
	return	zbuf

def func(exp):
	
	img_out_dir = os.path.join(dir_out, exp)
	if not os.path.exists(img_out_dir):
		os.makedirs(img_out_dir, exist_ok=True)
	matting_out_dir = os.path.join(dir_matting_out, exp)
	if not os.path.exists(matting_out_dir):
		os.makedirs(matting_out_dir, exist_ok=True)

	Ks, w2cs, dists, IDs = load_json(os.path.join(dir_img, exp,'params.json'))
	v_fac,tri= load_obj(os.path.join(dir_model, exp+'.obj'))
	# face to world
	f2w = load_sRt('facescape_Rt_scale_dict.json', int(ID), exp)
	f2w = np.concatenate([f2w[:3,:],[[0,0,0,1]]], 0)
	v_wrd = v_fac.dot(f2w[:3,:3].T) + f2w[:3,3:4].T
	# renderme-flame to facescape
	r2f = np.array([ \
		[1008.7127,  -7.1758,   5.5975, 0.5087], \
		[   7.8508, 999.9297,-132.9023, 4.8736], \
		[  -4.6031, 132.9404, 999.9449, 7.2580], \
		[0,0,0,1]], f2w.dtype)
	v_flm = ((v_wrd	-f2w[:3,3:4].T).dot(np.linalg.inv(f2w[:3,:3]).T) \
			-r2f[:3,3:4].T).dot(np.linalg.inv(r2f[:3,:3]).T)
	v_ren = v_flm.dot(r2w[:3,:3].T) + r2w[:3,3:4].T

	Ks_new = {}; w2cs_new = {}; near = float('inf'); far = 0

	for cam in IDs:
		i = IDs.index(int(cam))
		bgr = cv2.imread(os.path.join(dir_img, exp,'%d.jpg' % cam))
		K, w2c, dist = Ks[i],np.concatenate([w2cs[i][:3,:],[[0,0,0,1]]],0),dists[i]
		r2c = w2c.dot(f2w).dot(r2f).dot(np.linalg.inv(r2w))
		v_cam = v_wrd.dot(w2c[:3,:3].T) + w2c[:3,3:4].T
		scale = np.linalg.det(r2c[:3,:3])**(1./3)
		r2c[:3,:] /= scale

		metric  = [r2c[2,:3].dot(w2c[2,:3]) for w2c in renderme_w2cs]
		nearest = np.argmax(metric)
		renderme_K, renderme_w2c= renderme_cam[renderme_views[nearest]]['K'], \
					renderme_w2cs[nearest]
		Rz = renderme_w2c.dot(np.linalg.inv(r2c))
		v_cam = v_ren.dot(r2c[:3,:3].T) + r2c[:3,3:4].T
		near = min(near, v_cam[:,2].min())
		far  = max(far,  v_cam[:,2].max())
		zmean = r2c[2,3] * 9.0
		
		roll = np.arctan2(Rz[1,0]-Rz[0,1], Rz[0,0]+Rz[1,1])
		tx = Rz[0,3] / (r2c[2,3] + zmean)
		ty = Rz[1,3] / (r2c[2,3] + zmean)
		Rz = np.array([	[np.cos(roll),-np.sin(roll),tx], \
				[np.sin(roll), np.cos(roll),ty], \
				[0,0,1]], Rz.dtype)
		A = renderme_K.dot(Rz[:3,:3]).dot(np.linalg.inv(K))
		K_new  = renderme_K.copy()
		K_new[:2,2] += K_new[:2,:2].dot([tx,ty])
		renderme_bgr = np.zeros((2048,2448,3))
		h,w = bgr.shape[:2]
		bgr = cv2.warpAffine(bgr, A[:2,:], \
			(renderme_bgr.shape[1],renderme_bgr.shape[0]))
		Rz = np.array([	[Rz[0,0],Rz[0,1],0,0], \
				[Rz[1,0],Rz[1,1],0,0], \
				[0,0,1,0],[0,0,0,1]])
		w2c_new= Rz.dot(r2c)
		cv2.imwrite(os.path.join(img_out_dir, '%d.jpg' % cam), bgr)

		zbuf = np.ones([h, w], dtype = np.float32) * float('inf')
		zbuf = zbuffer(v_cam.dot(K.T), tri, zbuf, \
			persp = True, double_face = True)
		# zbuf = zbuffer_fast(v_cam.dot(K.T), tri, zbuf)
		mask =(zbuf < float('inf')).astype(np.uint8) * 255
		mask = cv2.warpAffine(mask, A[:2,:], \
			(renderme_bgr.shape[1],renderme_bgr.shape[0]))
		mask = mask.reshape(mask.shape[0],mask.shape[1],-1)[:,:,-1:]
		cv2.imwrite(os.path.join(matting_out_dir, '%d.png' % cam), mask)

		Ks_new[cam] = K_new
		w2cs_new[cam]=w2c_new
	
	with open(os.path.join(dir_img, exp,'params.json'), 'r') as fp:
		data = json.load(fp)
	for cam in IDs:
		data[f'{cam}_K'] = Ks_new[cam].tolist()
		data[f'{cam}_Rt'] = w2cs_new[cam].tolist()
		data[f'{cam}_width'] = renderme_bgr.shape[1]
		data[f'{cam}_height'] = renderme_bgr.shape[0]
	with open(os.path.join(img_out_dir,'params.json'), 'w') as fp:
		json.dump(data, fp)

base_dir = "./facescape" # TODO: input your path to dataset
ID = sys.argv[1]
print(ID)
dir_img = os.path.join(base_dir, 'multi_view_data/fsmview_trainset', ID)
dir_model = os.path.join(base_dir, 'tu_model', ID, 'models_reg')
dir_out = os.path.join(base_dir, 'multi_view_data/images_aligned', ID)
dir_matting_out = os.path.join(base_dir, 'multi_view_data/matting_aligned', ID)
if not os.path.exists(dir_img) or not os.path.exists(dir_model):
	exit(0)

# prepare general data
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
renderme_w2cs= np.array([ \
	np.linalg.inv(np.concatenate([ \
	renderme_cam[_]['RT'][:3,:], [[0,0,0,1]]], 0)) \
	for _ in renderme_views])

exps = sorted(os.listdir(dir_img))

use_multiprocess = False
if use_multiprocess:
	import multiprocessing
	max_processes = 4
	pool = multiprocessing.Pool(processes=max_processes)
	pbar = tqdm(total=len(exps))
	pbar.set_description(ID)
	update = lambda *args: pbar.update()

for exp in tqdm(exps):
	if use_multiprocess:
		pool.apply_async(func, (exp,), callback=update)
	else:
		func(exp)
if use_multiprocess:
	pool.close()
	pool.join()
