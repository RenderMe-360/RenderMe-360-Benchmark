# Generalizable Novel View Synthesis

## Setting
We provide three testing settings for the novel view synthesis task, including Protocal-1, Protocal-2 and In-Domain setting.

The structure of `data/` is as follows:

```
data/
├── protocal_1
│   ├── raw
│   ├── annotations
│   └── pretrained_models
├── protocal_2
│   ├── raw
│   ├── annotations
│   └── pretrained_models
├── in_domain
│   ├── raw
│   ├── annotations
│   └── pretrained_models
```



## IBRNet

1. Git clone from original repo


```
git clone https://github.com/yangyangwithgnu/IBRNet.git
```

2. Download pre-trained models and save the model to `{setting}/pretrained_models/IBRNet`

3. Copy the config and data loader files from `IBRNet/` to the original repo `ibrnet`.

4. Change the necessary path in the config file

5. Run the `eval` code

```
python eval.py --config ../configs/renderme.txt --eval_scenes {your eval scenes} --ckpt_path {your path}/latest.pth
```




## Keypoint-NeRF

✨ To Be Released....

## Vision-NeRF


✨ To Be Released....
