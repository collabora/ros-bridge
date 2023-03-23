import argparse
import copy
import os
import warnings

import mmcv
import numpy as np
import torch
import time
from pyquaternion import Quaternion

from torchpack.utils.config import configs
# from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval
from mmdet3d.core import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default="viz_infer")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    # dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    print()
    pipeline = Compose(cfg.data.test.pipeline)
    box_type_3d, box_mode_3d = get_box_type('LiDAR')
        
    # build data
    info = get_data(*get_sample_data())
    info = build_data(info)
    sample = prepare_sample(info, pipeline, box_mode_3d, box_type_3d)
    
    # run inference
        # build the model and load checkpoint

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = cfg.object_classes
    model.cuda()
    model.eval()
    # for k, v in sample.items():
    #     if k not in ['img', 'points', 'lidar_aug_matrix', 'metas', 'lidar2image', 'img_aug_matrix', 'lidar2ego', 'lidar2camera', 'camera2ego', 'camera_intrinsics']:
    #         print(k, v.shape, v)
    # xx
    with torch.no_grad():
        outputs = model(**sample)
    
    # visualize
    bboxes = outputs[0]["boxes_3d"].tensor.numpy()
    scores = outputs[0]["scores_3d"].numpy()
    labels = outputs[0]["labels_3d"].numpy()

    if args.bbox_classes is not None:
        indices = np.isin(labels, args.bbox_classes)
        bboxes = bboxes[indices]
        scores = scores[indices]
        labels = labels[indices]
    
    if args.bbox_score is not None:
        print("filtering bboxes")
        indices = scores >= args.bbox_score
        bboxes = bboxes[indices]
        scores = scores[indices]
        labels = labels[indices]
    bboxes[..., 2] -= bboxes[..., 5] / 2
    bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

    metas = sample["metas"][0]
    name = "{}-{}".format(str(time.time()), str(1))
    if "img" in sample:
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
    if "points" in sample:
            lidar = sample["points"][0].cpu().numpy()
            print(lidar.shape)
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

    # print(sample)

def get_sensor_calib_data():
    sensor_data = {
        'CAM_BACK': {
            "translation": [
                -2.0,
                -0.0,
                2.8
            ],
            "rotation": [
                0.5,
                -0.5,
                -0.5,
                0.5
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_FRONT': {
            "translation": [
                1.5,
                -0.0,
                2.8
            ],
            "rotation": [
                0.5,
                -0.5,
                0.5,
                -0.5
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_FRONT_LEFT': {
            "translation": [
                1.3,
                0.4,
                2.8
            ],
            "rotation": [
                0.6743797,
                -0.6743797,
                0.2126311,
                -0.2126311
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_FRONT_RIGHT': {
            "translation": [
                1.3,
                -0.4,
                2.8
            ],
            "rotation": [
                0.2126311,
                -0.2126311,
                0.6743797,
                -0.6743797
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_BACK_LEFT': {
            "translation": [
                -0.85,
                0.4,
                2.8
            ],
            "rotation": [
                0.6963642,
                -0.6963642,
                -0.1227878,
                0.1227878
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_BACK_RIGHT': {
            "translation": [
                -0.85,
                -0.4,
                2.8
            ],
            "rotation": [
                -0.1227878,
                0.1227878,
                0.6963642,
                -0.6963642
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'LIDAR_TOP': {
            "translation": [
                0.94,
                -0.0,
                2.8
            ],
            "rotation": [
                1.0,
                0.0,
                0.0,
                0.0
            ],
            "camera_intrinsic": []
        }
    }

    ego_pose = {
        "translation": [
            88.38433074951172,
            -88.25868225097656,
            0.0
        ],
        "rotation": [
            0.7076030964832982,
            6.146967038796277e-06,
            5.149090818303537e-06,
            -0.7066101172378937
        ],
    }

    return sensor_data, ego_pose

def get_sample_data():
    vehicle = 'model3'
    date = '01-08-2023'
    timestamp = '1673120610692'
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    # read images
    images = {}
    for cam in camera_types:
        img_path = f'{vehicle}-{date}__{cam}__{timestamp}.jpg'
        images[cam] = os.path.join(f'./data/nuscenes/samples/{cam}', img_path)

    # read lidar points
    points_file = os.path.join(f'./data/nuscenes/samples/LIDAR_TOP', f'{vehicle}-{date}__LIDAR_TOP__{timestamp}.pcd.bin')
    # read cam intrinsics
    sensor_calib_data, ego_pose  = get_sensor_calib_data()
    all_cam_intrinsics = {}
    cam_translations = {}
    cam_rotations = {}
    for cam in camera_types:
        all_cam_intrinsics[cam] = sensor_calib_data[cam]['camera_intrinsic']
        cam_translations[cam] = sensor_calib_data[cam]['translation']
        cam_rotations[cam] = sensor_calib_data[cam]['rotation']

    return images, points_file, all_cam_intrinsics, sensor_calib_data['LIDAR_TOP']['translation'], sensor_calib_data['LIDAR_TOP']['rotation'], \
        ego_pose['translation'], ego_pose['rotation'], cam_translations, cam_rotations


def get_data(
    images, lidar_path, all_cam_intrinsics, lidar2ego_translation, lidar2ego_rotation,
    ego2global_translation, ego2global_rotation, cam_translations, cam_rotations
    ):
    # print(points.shape)
    # print(all_cam_intrinsics, lidar2ego_translation, lidar2ego_rotation,
    # ego2global_translation, ego2global_rotation, cam_translations, cam_rotations)
    info = {
        "lidar_path": lidar_path,
        "cams": dict(),
        "lidar2ego_translation": lidar2ego_translation,
        "lidar2ego_rotation": lidar2ego_rotation,
        "ego2global_translation": ego2global_translation,
        "ego2global_rotation": ego2global_rotation,
    }


    l2e_r = info["lidar2ego_rotation"]
    l2e_t = info["lidar2ego_translation"]
    e2g_r = info["ego2global_rotation"]
    e2g_t = info["ego2global_translation"]
  
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain 6 image's information per frame
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    for cam in camera_types:
        cam_intrinsics = all_cam_intrinsics[cam]
        cam_info = obtain_sensor2top(
            images[cam], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, e2g_r, cam, cam_translations, cam_rotations
        )

        cam_info.update(camera_intrinsics=cam_intrinsics)
        info["cams"].update({cam: cam_info})
    info["sweeps"] = []
    return info


def obtain_sensor2top(
    data, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, e2g_r, sensor_type="lidar", cam_translations=None, cam_rotations=None, lidar_rotation=None
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    if sensor_type!= "lidar":
        translation = cam_translations[sensor_type],
        rotation = cam_rotations[sensor_type]
    else:
        translation = l2e_t
        rotation = lidar_rotation
    sweep = {
        "data_path": data,
        "type": sensor_type,
        "sensor2ego_translation": translation[0],
        "sensor2ego_rotation": rotation,
        "ego2global_translation": e2g_t,
        "ego2global_rotation": e2g_r,
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def build_data(info, use_camera=True):
    data = dict(
        lidar_path=info["lidar_path"],
        sweeps=info["sweeps"],
    )

    # ego to global transform
    ego2global = np.eye(4).astype(np.float32)
    ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = info["ego2global_translation"]
    data["ego2global"] = ego2global

    # lidar to ego transform
    lidar2ego = np.eye(4).astype(np.float32)
    lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = info["lidar2ego_translation"]
    data["lidar2ego"] = lidar2ego

    if use_camera:
        data["image_paths"] = []
        data["lidar2camera"] = []
        data["lidar2image"] = []
        data["camera2ego"] = []
        data["camera_intrinsics"] = []
        data["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            data["image_paths"].append(camera_info["data_path"])
            
            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            data["lidar2camera"].append(lidar2camera_rt.T)
            
            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            data["camera_intrinsics"].append(camera_intrinsics)
            
            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            data["lidar2image"].append(lidar2image)
            
            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            # print(camera_info["sensor2ego_rotation"])
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"][0]
            data["camera2ego"].append(camera2ego)
            
            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            data["camera2lidar"].append(camera2lidar)
        
    return data


def prepare_sample(results, pipeline, box_mode_3d, box_type_3d):
    results["img_fields"] = []
    results["bbox3d_fields"] = []
    results["pts_mask_fields"] = []
    results["pts_seg_fields"] = []
    results["bbox_fields"] = []
    results["mask_fields"] = []
    results["seg_fields"] = []
    results["box_type_3d"] = box_type_3d
    results["box_mode_3d"] = box_mode_3d

    example = pipeline(results)
    sample = {}
    for k, v in example.items():
        print(k, len(v))
        if k == 'points':
            sample[k] = [torch.FloatTensor(example[k].data).cuda()]
        else:
            try:
                sample[k] = torch.unsqueeze(example[k].data, dim=0).cuda()
            except Exception as e:
               sample[k] = [example[k].data]
    
    return sample


if __name__=="__main__":
    main()