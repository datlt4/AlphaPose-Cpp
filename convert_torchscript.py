import argparse
import torch
import cv2
import numpy as np

from alphapose.models import builder
from alphapose.utils.config import update_config
print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform YOLO weights to ONNX.')
    parser.add_argument('--cfg', type=str, help='alphapose cfg file')
    parser.add_argument('--pth', type=str, help='alphapose weights file')
    args = parser.parse_args()

    cfg = update_config(args.cfg)
    
    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    pose_model.load_state_dict(torch.load(args.pth, map_location=device))
    pose_model.to(device)
    pose_model.eval()

    # Create sample
    image = np.random.randint(0, 255, (3, 256, 192), dtype=np.uint8)
    img = torch.from_numpy(image).float()
    img /= 255
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)
    imgs = img.unsqueeze(0)
    imgs = imgs.to(device)

    # Convert by jit::trace or jit::script
    traced_model = torch.jit.trace(pose_model, imgs)
    # traced_model = torch.jit.script(pose_model)

    # Save torchscript model
    traced_model.save(args.pth.replace(".pth", ".jit"))


"model-zoo/fast_pose_res152/256x192_res152_lr1e-3_1x-duc.yaml"
"model-zoo/fast_pose_res50/256x192_res50_lr1e-3_1x.yaml"

"model-zoo/fast_pose_res152/fast_421_res152_256x192.pth"
"model-zoo/fast_pose_res50/fast_res50_256x192.pth"