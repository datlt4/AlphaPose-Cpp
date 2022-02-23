# AlphaPose-Cpp

## Convert model to TensorRT

1. __Clone Github repository__

```bash
git clone --recursive https://github.com/LuongTanDat/AlphaPose-Cpp.git -b tensorrt
cd AlphaPose-Cpp
export WORKDIR=$(pwd)
```

2. __Convert from pretrain model__

- Install AlphaPose python as following [repo](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md#code-installation).

- Download pretrained model and config file as following [path](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md).

__[NOTE]:__ my code only support 17-keypoint model currently. If you want process more keypoint, you need to modify code in `AlphaPose::postprocess` method.

- Code convert to `Onnx`. [Reference](https://github.com/myl980/AlphaPose2Trt/blob/main/fastpose2onnxDynamic.py)

```bash
python convert_onnx.py --cfg "model-zoo/fast_pose_res50/256x192_res50_lr1e-3_1x.yaml" --pth "model-zoo/fast_pose_res50/fast_res50_256x192.pth" --out model-zoo/fast_pose_res50/fast_res50_256x192_dynamic.onnx --dynamic
# python convert_onnx.py --cfg "model-zoo/fast_pose_res50/256x192_res50_lr1e-3_1x.yaml" --pth "model-zoo/fast_pose_res50/fast_res50_256x192.pth" --out model-zoo/fast_pose_res50/fast_res50_256x192.onnx
```

3. Download my converted model by using `dvc`

- Install `dvc`

```bash
$(which python) -m pip install dvc dvc[gdrive]
dvc pull
```

## Build

### Build `trtexec` application

```bash
cd ${WORKDIR}/TrtExec/build
cmake ..
cmake --build . --config Release
./Trtexec \
    --onnx ../../model-zoo/fast_pose_res50/fast_res50_256x192_dynamic.onnx \
    --engine ../../model-zoo/fast_pose_res50/fast_res50_256x192_fp16_dynamic.engine \
    --inputName "input" \
    --minShape 1x3x256x192 \
    --optShape 8x3x256x192 \
    --maxShape 32x3x256x192 \
    --workspace 1024
```

### Build test application

```bash
cd ${WORKDIR}/build
cmake ..
cmake --build . --config Release
./alApp
```