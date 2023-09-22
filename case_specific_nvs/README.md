# Case-specific NVS

This case-specific track refers to the setting of training on a single head with multi-view images, which originates from NeRF’s de facto setting, to evaluate the robustness of static multi-view head reconstruction.


# Reimplementation of instant-ngp

### Preparation
 
1. Clone the official repo.
```sh
$ git clone --recursive https://github.com/nvlabs/instant-ngp
```
2. Replace the following files with the same place. The official ngp code defaults to single-camera setups and does not support multi-view cameras with different camera intrinsics. We adjust the code to fit multi-view setting.
```sh
$ mv case_specific_nvs/instant-ngp/testbed.h instant-ngp/include/neural-graphics-primitives/testbed.h
$ mv case_specific_nvs/instant-ngp/testbed.cu instant-ngp/src/testbed.cu
$ mv case_specific_nvs/instant-ngp/python_api.cu instant-ngp/src/python_api.cu
```
3. Compile the repo following the official steps.
4. Download and unzip the benchmark dataset from `benchmark_data/data_NVS_instant-ngp.zip` and move it to the root path.
5. We also provided our pre-trained checkpoints (sampled checpoints, due to the large model size) of instant-ngp [here](https://drive.google.com/drive/folders/11qK2CEevV2Oi5Jm-sFv913CNyrYXqY5y?usp=sharing).

### Training

1. Move the scripts file.
```sh
$ mv case_specific_nvs/instant-ngp/scripts/* instant-ngp/rscripts/
$ mv case_specific_nvs/instant-ngp/run.sh instant-ngp/run.sh
$ mv case_specific_nvs/instant-ngp/calculate_metric.py instant-ngp/calculate_metric.py
```
2. Modify the scripts - run.sh
- STEPS=30000 means the number of training step is 30000
- SINGLE=0 means do not use single training mode (train one ngp model)
- MULTI=1  means use multiple training mode (train all 40 models parallel, need 40 gpus)
- TRAIN=1 means run the train mode (set to 1)
- RENDER=0 means do not run the render mode (set to 0)

3. Start the training
```sh
$ bash run.sh
```

### Evaluation
1. Modify the scripts - run.sh
- TRAIN=0 
- RENDER=1

2. Start the evaluation/rendering
```sh
$ bash run.sh
$ python calculate_metric.py
```

## Reimplementation of neuralnolumes

### Preparation

1. Clone the official repo.
```sh
$ git clone https://github.com/facebookresearch/neuralvolumes.git
```

2. Build up the python environment according to the official repo.

3. Move/Replace the scripts files.
```sh
$ cp -r case_specific_nvs/neuralnolumes/*.py neuralnolumes/
$ cp -r case_specific_nvs/neuralnolumes/models neuralnolumes/models
$ cp -r case_specific_nvs/neuralnolumes/eval neuralnolumes/eval
$ cp -r case_specific_nvs/neuralnolumes/runs neuralnolumes/
$ cp -r case_specific_nvs/neuralnolumes/data/*.py neuralnolumes/data/
```

4. Download and unzip the benchmark dataset from `benchmark_data/data_NVS_mvp_nv.tar` and move it to the root path.
5. We also provided our pre-trained checkpoints (sampled checpoints, due to the large model size)  of neuralvolumes [here](https://drive.google.com/drive/folders/1IqJrb2VKNrVKOgMhzjPXtG9bvu0n-3nj?usp=sharing). 

### Training
1. start the training
```sh
$ bash runs/train.sh
```

### Evaluation
1. start the evaluation
```sh
$ bash runs/eval.sh
```

2. calculate the metric
```sh
$ python calculate_metric.py
```

## Reimplementation of Mixture of Volumetric Primitives

### Training/Testing

1. Git clone the original repository of [MVP](https://github.com/facebookresearch/mvp) and create the virtual environment following the instruction.
2. Build up the python environment according to the official repo.
3. Fit the code with provided dataloader and config files.
4. Download and unzip the benchmark dataset from `benchmark_data/data_NVS_mvp_nv.tar` and move it to the root path.
5. We also provided our pre-trained checkpoints (sampled checpoints, due to the large model size)  of MVP [here](https://drive.google.com/drive/folders/1dqSFkUCrPxcPOQiRxwcN61dRPdWrN1Fs?usp=sharing). 
6. Start training/testing.


## Citation
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