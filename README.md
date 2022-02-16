# AlphaPose-Cpp

## Convert model to TensorRT

1. __Convert from pretrain model__

- Install AlphaPose python as following [repo](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md#code-installation).

- Download pretrained model and config file as following [path](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md).

__[NOTE]:__ my code only support 17-keypoint model currently. If you want process more keypoint, you need to modify code in `AlphaPose::postprocess` method.

- Code convert to `Onnx`. [Reference](https://github.com/myl980/AlphaPose2Trt/blob/main/fastpose2onnxDynamic.py)

```bash
python convert_onnx.py --cfg "model-zoo/fast_pose_res50/256x192_res50_lr1e-3_1x.yaml" --pth "model-zoo/fast_pose_res50/fast_res50_256x192.pth" --out model-zoo/fast_pose_res50/fast_res50_256x192_dynamic.onnx --dynamic
# python convert_onnx.py --cfg "model-zoo/fast_pose_res50/256x192_res50_lr1e-3_1x.yaml" --pth "model-zoo/fast_pose_res50/fast_res50_256x192.pth" --out model-zoo/fast_pose_res50/fast_res50_256x192.onnx
```

2. Download my converted model by using `dvc`

- Install `dvc`

```bash
$(which python) -m pip install dvc dvc[gdrive]
dvc pull
```

## Build

### Build `trtexec` application

```bash
mkdir -p build && cd build
cmake -DBUILD_TRTEXEC=ON ..
cmake --build . --config Release
./alTrtexec --onnx ../model-zoo/fast_pose_res50/fast_res50_256x192_dynamic.onnx \
    --engine ../model-zoo/fast_pose_res50/fast_res50_256x192_fp16_dynamic.engine \
    --minBatchSize 1 --optBatchSize 8 --maxBatchSize 32 --dynamic
```

### Build test application

```bash
mkdir -p build && cd build
cmake -DBUILD_TRTEXEC=OFF ..
cmake --build . --config Release
./alApp
```