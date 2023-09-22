# Novel Expression Synthesis

This task refers to the setting of reconstructing a 4D facial avatar based on monocular video sequences.

## Reimplementation of IM Avatar

### Preparation

1. Clone the official repo.
```sh
$ git clone https://github.com/zhengyuf/IMavatar
```

2. Build up the python environment according to the official repo.

3. Move/Replace the scripts files.
```sh
$ mv novel_expression_synthesis/IMavatar/*.sh IMavatar/
$ mv novel_expression_synthesis/IMavatar/*.py IMavatar/
$ mv novel_expression_synthesis/IMavatar/scripts IMavatar/scripts
$ mv novel_expression_synthesis/IMavatar/datasets IMavatar/datasets
```

4. Download and unzip the benchmark dataset from `benchmark_data/data_NES_imavatar_pointavatar.tar` and move it to the root path.
```sh
$ mv [path to dataset] IMavatar/data
```
5. We also provided our pre-trained checkpoints of imavatar [here](https://drive.google.com/drive/folders/1ppL-v5mcC2y4rP6JosI_LfTqUPYtmkH1?usp=sharing).


### Training
1. start the training
```sh
$ bash train.sh
```

### Evaluation
1. evaluation 
```sh
$ bash eval.sh
```

2. calculate metric
```sh
$ python calculate_metric.py
```

## Reimplementation of Point Avatar

### Preparation

1. Clone the official repo.
```sh
$ git clone https://github.com/zhengyuf/PointAvatar
```

2. Build up the python environment according to the official repo.

3. Move/Replace the scripts files.
```sh
$ mv novel_expression_synthesis/pointavatar/*.sh pointavatar/
$ mv novel_expression_synthesis/pointavatar/calculate_metric.py pointavatar/calculate_metric.py
$ mv novel_expression_synthesis/pointavatar/scripts pointavatar/code/scripts
$ mv novel_expression_synthesis/pointavatar/datasets pointavatar/code/datasets
```

4. Download the FLAME model from [here](https://drive.google.com/file/d/1A9kLFmNeag63LZ_iyxQanxU4fgEvVFOM/view?usp=sharing), and put it to pointavatar/code/flame/FLAME2020/generic_model.pkl.

5. Download and unzip the benchmark dataset from `benchamrk_data/data_NES_imavatar_pointavatar.tar` and move it to the root path.
```sh
$ mv [path to dataset] pointavatar/data
```
6. We also provided our pre-trained checkpoints of pointavatar [here](https://drive.google.com/drive/folders/1TXswgi-hPqFUH4pmxt_DvCY0gVM8D5n1?usp=sharing).


### Training
1. start the training
```sh
$ bash train.sh
```

### Evaluation
1. evaluation 
```sh
$ bash eval.sh
```

2. calculate metric
```sh
$ python calculate_metric.py
```

## Citation

```
@InProceedings{Gafni_2021_CVPR,
  year={2021},
    author    = {Gafni, Guy and Thies, Justus and Zollh{\"o}fer, Michael and Nie{\ss}ner, Matthias},
    title     = {Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction},
    booktitle = {CVPR}
}
```
```
@inproceedings{zheng2022avatar,
  year={2022},
  title={Im avatar: Implicit morphable head avatars from videos},
  author={Zheng, Yufeng and Abrevaya, Victoria Fern{\'a}ndez and B{\"u}hler, Marcel C and Chen, Xu and Black, Michael J and Hilliges, Otmar},
  booktitle={CVPR}
}
```
```
@article{zheng2022pointavatar,
  year={2022},
  title={PointAvatar: Deformable Point-based Head Avatars from Videos},
  author={Zheng, Yufeng and Yifan, Wang and Wetzstein, Gordon and Black, Michael J and Hilliges, Otmar},
  journal={arXiv}
}
```