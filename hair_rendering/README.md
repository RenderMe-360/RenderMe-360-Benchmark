# Hair Rendering

This task refers to the setting of modeling accurate hair appearance across changes of viewpoints or dynamic motions.

## Reimplementation of Neural-Scene-Flow-Fields

### Preparation
 
1. Clone the official repo.
```sh
$ git clone https://github.com/zhengqili/Neural-Scene-Flow-Fields
```
2. Move/Replace the following files. 
```sh
$ cp -r hair_rendering/nsff/nsff_exp Neural-Scene-Flow-Fields/nsff_exp
$ cp hair_rendering/nsff/calculate_metric.py Neural-Scene-Flow-Fields/calculate_metric.py
$ cp hair_rendering/nsff/render.sh Neural-Scene-Flow-Fields/
$ cp hair_rendering/nsff/train.sh Neural-Scene-Flow-Fields/
```
3. Build up the runing environment.

4. Download and unzip the benchmark dataset from `benchamrk_data/data_hairrendering_nsff_nrnerf.tar` and move it to the root path.
```sh
$ mv [path to dataset] Neural-Scene-Flow-Fields/
```
5. We also provided our pre-trained checkpoints of NSFF [here](https://drive.google.com/drive/folders/1ICVVlIqVXOPCn-KhaPuXUn7ICWCNNqkT?usp=sharing).


### Training

1. start the training
```sh
$ bash train.sh
```

### Evaluation

1. start the evaluation
```sh
$ bash metric_timeinter.sh
```

2. calculate metric
```sh
$ python metrics_calculation.py
```

## Reimplementation of Nonrigid-NeRF

### Preparation
 
1. Clone the official repo.
```sh
$ git clone https://github.com/facebookresearch/nonrigid_nerf
```

2. Move/Replace the following files. 
```sh
$ cp -r hair_rendering/nrnerf/*.py nonrigid_nerf/
$ cp -r hair_rendering/nrnerf/*.sh nonrigid_nerf/
```

3. Build up the runing environment.

4. Download and unzip the benchmark dataset from `benchamrk_data/data_hairrendering_nsff_nrnerf.tar` and move it to the root path.
```sh
$ mv [path to dataset] nonrigid_nerf/
```
5. We also provided our pre-trained checkpoints of NRNeRF [here](https://drive.google.com/drive/folders/1zVe_YIEmtOf4DCKrquJrGREQNN7C28nL?usp=sharing).


### Training

1. start the training
```sh
$ bash train.sh
```

### Evaluation

1. start the evaluation
```sh
$ bash render.sh
```

2. calculate metric
```sh
$ python metrics_calculation.py
```

## Citation

```
 @inproceedings{2021nsff,
   year={2021},
  title={Neural scene flow fields for space-time view synthesis of dynamic scenes},
  author={Li, Zhengqi and Niklaus, Simon and Snavely, Noah and Wang, Oliver},
  booktitle={CVPR}
}
```

```
@inproceedings{2021nrnerf,
  year={2021},
  title={Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video},
  author={Tretschk, Edgar and Tewari, Ayush and Golyanik, Vladislav and Zollh{\"o}fer, Michael and Lassner, Christoph and Theobalt, Christian},
  booktitle={ICCV}
}
```

```
@article{muller2022instant,
  year={2022},
  title={Instant neural graphics primitives with a multiresolution hash encoding},
  author={M{\"u}ller, Thomas and Evans, Alex and Schied, Christoph and Keller, Alexander},
  journal={TOG}
}
```
```
@article{Lombardi21,
year={2021},
author = {Lombardi, Stephen and Simon, Tomas and Schwartz, Gabriel and Zollhoefer, Michael and Sheikh, Yaser and Saragih, Jason},
title = {Mixture of Volumetric Primitives for Efficient Neural Rendering},
journal = {TOG}
}
```
```
@article{wang2021neus,
  year={2021},
  title={Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction},
  author={Wang, Peng and Liu, Lingjie and Liu, Yuan and Theobalt, Christian and Komura, Taku and Wang, Wenping},
  journal={arXiv}
}
```
```
@article{lombardi2019neural,
  year={2019},
  title={Neural volumes: Learning dynamic renderable volumes from images},
  author={Lombardi, Stephen and Simon, Tomas and Saragih, Jason and Schwartz, Gabriel and Lehrmann, Andreas and Sheikh, Yaser},
  journal={arXiv}
}
```