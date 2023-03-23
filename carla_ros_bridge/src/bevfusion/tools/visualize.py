import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])
        for k, v in data.items():
            if k not in ['img', 'points', 'lidar_aug_matrix', 'metas', 'lidar2image', 'img_aug_matrix', 'lidar2ego', 'lidar2camera', 'camera2lidar']:
                print(k, v)
        xx
        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()
            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            print(bboxes.dims)
            print(bboxes)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )
        xx
# camera2ego DataContainer([tensor([[[[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  1.5000e+00],
#           [-1.0000e+00,  0.0000e+00,  0.0000e+00, -0.0000e+00],
#           [ 0.0000e+00, -1.0000e+00,  0.0000e+00,  2.8000e+00],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-8.1915e-01,  1.4101e-18,  5.7358e-01,  1.3000e+00],
#           [-5.7358e-01,  1.1460e-17, -8.1915e-01, -4.0000e-01],
#           [ 1.4101e-18, -1.0000e+00,  1.6296e-17,  2.8000e+00],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 8.1915e-01,  1.5979e-17,  5.7358e-01,  1.3000e+00],
#           [-5.7358e-01, -2.0326e-18,  8.1915e-01,  4.0000e-01],
#           [-1.1776e-17, -1.0000e+00,  2.0326e-18,  2.8000e+00],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 0.0000e+00,  0.0000e+00, -1.0000e+00, -2.0000e+00],
#           [ 1.0000e+00,  0.0000e+00,  0.0000e+00, -0.0000e+00],
#           [ 0.0000e+00, -1.0000e+00,  0.0000e+00,  2.8000e+00],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 9.3969e-01, -1.0201e-17, -3.4202e-01, -8.5000e-01],
#           [ 3.4202e-01,  2.7574e-17,  9.3969e-01,  4.0000e-01],
#           [ 3.6772e-18, -1.0000e+00,  2.7937e-17,  2.8000e+00],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-9.3969e-01, -1.0201e-17, -3.4202e-01, -8.5000e-01],
#           [ 3.4202e-01,  2.8011e-17, -9.3969e-01, -4.0000e-01],
#           [ 3.6772e-18, -1.0000e+00,  2.7500e-17,  2.8000e+00],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]])])
# lidar2ego DataContainer([tensor([[[1.0000, 0.0000, 0.0000, 0.9400],
#          [0.0000, 1.0000, 0.0000, -0.0000],
#          [0.0000, 0.0000, 1.0000, 2.8000],
#          [0.0000, 0.0000, 0.0000, 1.0000]]])])
# lidar2camera DataContainer([tensor([[[[ 4.7463e-20, -1.0000e+00, -8.4232e-23, -1.4211e-14],
#           [-1.6605e-21, -3.0395e-23, -1.0000e+00,  4.4409e-16],
#           [ 1.0000e+00, -2.9832e-19,  4.5849e-23, -5.6000e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-8.1915e-01, -5.7358e-01, -7.7249e-18,  6.5464e-02],
#           [ 8.1913e-18, -1.4157e-17, -1.0000e+00,  4.3548e-16],
#           [ 5.7358e-01, -8.1915e-01, -8.5803e-18, -5.3415e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 8.1915e-01, -5.7358e-01,  1.4255e-17, -6.5464e-02],
#           [-8.4839e-18,  8.4189e-18, -1.0000e+00,  4.4378e-16],
#           [ 5.7358e-01,  8.1915e-01,  7.4999e-18, -5.3415e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-4.7463e-20,  1.0000e+00,  8.4232e-23, -1.3954e-19],
#           [-1.6605e-21, -3.0395e-23, -1.0000e+00,  4.4408e-16],
#           [-1.0000e+00,  2.9832e-19, -4.5849e-23, -2.9400e+00],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 9.3969e-01,  3.4202e-01, -1.5546e-19,  1.5452e+00],
#           [-6.1014e-18,  2.7510e-17, -1.0000e+00,  4.2216e-16],
#           [-3.4202e-01,  9.3969e-01,  2.9400e-17, -9.8809e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-9.3969e-01,  3.4202e-01,  1.9167e-17, -1.5452e+00],
#           [-1.2859e-17, -2.4586e-17, -1.0000e+00,  4.1124e-16],
#           [-3.4202e-01, -9.3969e-01, -2.2832e-17, -9.8809e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]])])
# camera2lidar DataContainer([tensor([[[[-2.9832e-19,  4.5849e-23,  1.0000e+00,  5.6000e-01],
#           [-1.0000e+00,  8.4232e-23,  4.7463e-20, -1.4211e-14],
#           [ 3.0395e-23, -1.0000e+00, -1.6605e-21,  4.4409e-16],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-8.1915e-01,  1.4064e-18,  5.7358e-01,  3.6000e-01],
#           [-5.7358e-01,  1.1459e-17, -8.1915e-01, -4.0000e-01],
#           [ 1.4101e-18, -1.0000e+00,  1.6295e-17,  4.4409e-16],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 8.1915e-01,  1.5978e-17,  5.7358e-01,  3.6000e-01],
#           [-5.7358e-01, -2.0326e-18,  8.1915e-01,  4.0000e-01],
#           [-1.1778e-17, -1.0000e+00,  2.0302e-18,  4.4409e-16],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 2.9832e-19,  4.5849e-23, -1.0000e+00, -2.9400e+00],
#           [ 1.0000e+00,  8.4232e-23, -4.7463e-20,  0.0000e+00],
#           [-3.0395e-23, -1.0000e+00,  1.6605e-21,  4.4409e-16],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 9.3969e-01, -1.0202e-17, -3.4202e-01, -1.7900e+00],
#           [ 3.4202e-01,  2.7574e-17,  9.3969e-01,  4.0000e-01],
#           [ 3.6756e-18, -1.0000e+00,  2.7938e-17,  4.4409e-16],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-9.3969e-01, -1.0202e-17, -3.4202e-01, -1.7900e+00],
#           [ 3.4202e-01,  2.8011e-17, -9.3969e-01, -4.0000e-01],
#           [ 3.6745e-18, -1.0000e+00,  2.7501e-17,  4.4409e-16],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]])])
# lidar2image DataContainer([tensor([[[[ 8.0000e+02, -1.1425e+03, -5.9558e-20, -4.4800e+02],
#           [ 4.5000e+02, -1.3428e-16, -1.1425e+03, -2.5200e+02],
#           [ 1.0000e+00, -2.9832e-19,  4.5849e-23, -5.6000e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-4.7704e+02, -1.3106e+03, -1.5690e-14, -3.5252e+02],
#           [ 2.5811e+02, -3.6862e+02, -1.1425e+03, -2.4037e+02],
#           [ 5.7358e-01, -8.1915e-01, -8.5803e-18, -5.3415e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 1.3948e+03, -9.5367e-06,  2.2286e-14, -5.0211e+02],
#           [ 2.5811e+02,  3.6862e+02, -1.1425e+03, -2.4037e+02],
#           [ 5.7358e-01,  8.1915e-01,  7.4999e-18, -5.3415e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-8.0000e+02,  1.1425e+03,  5.9558e-20, -2.3520e+03],
#           [-4.5000e+02,  1.3421e-16, -1.1425e+03, -1.3230e+03],
#           [-1.0000e+00,  2.9832e-19, -4.5849e-23, -2.9400e+00],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[ 8.0000e+02,  1.1425e+03,  2.3343e-14,  9.7499e+02],
#           [-1.5391e+02,  4.2286e+02, -1.1425e+03, -4.4464e+02],
#           [-3.4202e-01,  9.3969e-01,  2.9400e-17, -9.8809e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

#          [[-1.3472e+03, -3.6099e+02,  3.6323e-15, -2.5559e+03],
#           [-1.5391e+02, -4.2286e+02, -1.1425e+03, -4.4464e+02],
#           [-3.4202e-01, -9.3969e-01, -2.2832e-17, -9.8809e-01],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]])])
# img_aug_matrix DataContainer([tensor([[[[   0.4800,    0.0000,    0.0000,  -32.0000],
#           [   0.0000,    0.4800,    0.0000, -176.0000],
#           [   0.0000,    0.0000,    1.0000,    0.0000],
#           [   0.0000,    0.0000,    0.0000,    1.0000]],

#          [[   0.4800,    0.0000,    0.0000,  -32.0000],
#           [   0.0000,    0.4800,    0.0000, -176.0000],
#           [   0.0000,    0.0000,    1.0000,    0.0000],
#           [   0.0000,    0.0000,    0.0000,    1.0000]],

#          [[   0.4800,    0.0000,    0.0000,  -32.0000],
#           [   0.0000,    0.4800,    0.0000, -176.0000],
#           [   0.0000,    0.0000,    1.0000,    0.0000],
#           [   0.0000,    0.0000,    0.0000,    1.0000]],

#          [[   0.4800,    0.0000,    0.0000,  -32.0000],
#           [   0.0000,    0.4800,    0.0000, -176.0000],
#           [   0.0000,    0.0000,    1.0000,    0.0000],
#           [   0.0000,    0.0000,    0.0000,    1.0000]],

#          [[   0.4800,    0.0000,    0.0000,  -32.0000],
#           [   0.0000,    0.4800,    0.0000, -176.0000],
#           [   0.0000,    0.0000,    1.0000,    0.0000],
#           [   0.0000,    0.0000,    0.0000,    1.0000]],

#          [[   0.4800,    0.0000,    0.0000,  -32.0000],
#           [   0.0000,    0.4800,    0.0000, -176.0000],
#           [   0.0000,    0.0000,    1.0000,    0.0000],
#           [   0.0000,    0.0000,    0.0000,    1.0000]]]])])
# lidar_aug_matrix DataContainer([tensor([[[1., 0., 0., 0.],
#          [0., 1., 0., 0.],
#          [0., 0., 1., 0.],
#          [0., 0., 0., 1.]]])])
# metas DataContainer([[{'filename': ['./data/nuscenes/samples/CAM_FRONT/model3-01-08-2023__CAM_FRONT__1673120610692.jpg', './data/nuscenes/samples/CAM_FRONT_RIGHT/model3-01-08-2023__CAM_FRONT_RIGHT__1673120610692.jpg', './data/nuscenes/samples/CAM_FRONT_LEFT/model3-01-08-2023__CAM_FRONT_LEFT__1673120610692.jpg', './data/nuscenes/samples/CAM_BACK/model3-01-08-2023__CAM_BACK__1673120610692.jpg', './data/nuscenes/samples/CAM_BACK_LEFT/model3-01-08-2023__CAM_BACK_LEFT__1673120610692.jpg', './data/nuscenes/samples/CAM_BACK_RIGHT/model3-01-08-2023__CAM_BACK_RIGHT__1673120610692.jpg'], 'timestamp': 1673120610692, 'ori_shape': (1600, 900), 'img_shape': (1600, 900), 'lidar2image': [array([[ 8.0000000e+02, -1.1425184e+03, -5.9557708e-20, -4.4800000e+02],
#        [ 4.5000000e+02, -1.3427815e-16, -1.1425184e+03, -2.5200000e+02],
#        [ 1.0000000e+00, -2.9831871e-19,  4.5848855e-23, -5.6000000e-01],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
#       dtype=float32), array([[-4.7703516e+02, -1.3106433e+03, -1.5690086e-14, -3.5252466e+02],
#        [ 2.5810941e+02, -3.6861844e+02, -1.1425184e+03, -2.4036674e+02],
#        [ 5.7357645e-01, -8.1915206e-01, -8.5802791e-18, -5.3414834e-01],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
#       dtype=float32), array([[ 1.3947574e+03, -9.5367432e-06,  2.2286098e-14, -5.0211270e+02],
#        [ 2.5810941e+02,  3.6861844e+02, -1.1425184e+03, -2.4036674e+02],
#        [ 5.7357645e-01,  8.1915206e-01,  7.4998820e-18, -5.3414834e-01],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
#       dtype=float32), array([[-8.0000000e+02,  1.1425184e+03,  5.9557708e-20, -2.3520000e+03],
#        [-4.5000000e+02,  1.3420869e-16, -1.1425184e+03, -1.3230000e+03],
#        [-1.0000000e+00,  2.9831871e-19, -4.5848855e-23, -2.9400001e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
#       dtype=float32), array([[ 7.9999994e+02,  1.1425184e+03,  2.3342715e-14,  9.7499268e+02],
#        [-1.5390907e+02,  4.2286166e+02, -1.1425184e+03, -4.4464191e+02],
#        [-3.4202015e-01,  9.3969262e-01,  2.9400414e-17, -9.8809314e-01],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
#       dtype=float32), array([[-1.3472322e+03, -3.6098975e+02,  3.6323267e-15, -2.5559417e+03],
#        [-1.5390907e+02, -4.2286166e+02, -1.1425184e+03, -4.4464191e+02],
#        [-3.4202015e-01, -9.3969262e-01, -2.2832410e-17, -9.8809314e-01],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],

if __name__ == "__main__":
    main()
