import argparse
from sympy import arg
import torch
from alphapose.models import builder
from alphapose.utils.config import update_config


def saveONNX(model, filepath, c, h, w, dynamic=True):
    # 输入数据形状
    dummy_input = torch.randn(1, c, h, w, device='cuda')
    if dynamic:
        dynamic_ax = {'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
        torch.onnx.export(model, dummy_input, filepath, opset_version=10,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes=dynamic_ax)
    else:
        torch.onnx.export(model, dummy_input, filepath, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transform YOLO weights to ONNX.')
    parser.add_argument('--cfg', type=str, help='alphapose cfg file')
    parser.add_argument('--pth', type=str, help='alphapose weights file')
    parser.add_argument('--out', type=str, help='output onnx file')
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    cfg = update_config(args.cfg)
    print(cfg)
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    pose_model.load_state_dict(torch.load(args.pth))
    pose_model.eval()
    print(pose_model)
    pose_model = pose_model.cuda()

    saveONNX(pose_model, filepath=args.out, c=3,
             h=256, w=192, dynamic=args.dynamic)

# python convert_onnx.py --cfg "model-zoo/fast_pose_res50/256x192_res50_lr1e-3_1x.yaml" --pth "model-zoo/fast_pose_res50/fast_res50_256x192.pth" --out model-zoo/fast_pose_res50/fast_res50_256x192.pth 
# python convert_onnx.py --cfg "model-zoo/fast_pose_res50/256x192_res50_lr1e-3_1x.yaml" --pth "model-zoo/fast_pose_res50/fast_res50_256x192.pth" --out model-zoo/fast_pose_res50/fast_res50_256x192_dynamic.pth --dynamic