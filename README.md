# AlphaPose-Cpp

## Convert model to Torchscript

1. __Convert from pretrain model__

- Install AlphaPose python as following [repo](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md#code-installation).

- Download pretrained model and config file as following [path](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md).

__[NOTE]:__ my code only support 17-keypoint model currently. If you want process more keypoint, you need to modify code in `AlphaPose::postprocess` method.

- Code convert to Torchscript.

```bash
python convert_torchscript.py --cfg "model-zoo/fast_pose_res50/256x192_res50_lr1e-3_1x.yaml" --pth "model-zoo/fast_pose_res50/fast_res50_256x192.pth"
# python convert_torchscript.py --cfg "model-zoo/fast_pose_res152/256x192_res152_lr1e-3_1x-duc.yaml" --pth "model-zoo/fast_pose_res152/fast_421_res152_256x192.pth"
```

2. Download my converted model by using `dvc`

- Install `dvc`

```bash
$(which python) -m pip install dvc dvc[gdrive]
dvc pull
```