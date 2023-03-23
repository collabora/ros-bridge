#!/usr/bin/env python

import argparse
import copy
import os
import warnings

import mmcv
import numpy as np
import torch
import time
import json
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


def read_json(filename):
    """Reads json file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist.")
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def get_data(
    images, points, all_cam_intrinsics, lidar2ego_translation, lidar2ego_rotation,
    ego2global_translation, ego2global_rotation, cam_translations, cam_rotations
    ):

    info = {
        # "lidar_path": lidar_path,
        "points": points,
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
        # "data_path": data,
        "image": data,
        "type": sensor_type,
        "sensor2ego_translation": translation,
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
        # lidar_path=info["lidar_path"],
        points=info["points"],
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
    imgs = {}

    if use_camera:
        # data["image_paths"] = []
        data['images'] = []
        data["lidar2camera"] = []
        data["lidar2image"] = []
        data["camera2ego"] = []
        data["camera_intrinsics"] = []
        data["camera2lidar"] = []

        for cm_id, camera_info in info["cams"].items():
            # data["image_paths"].append(camera_info["data_path"])
            data["images"].append(camera_info["image"])
            imgs[cm_id] = camera_info["image"]
            
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
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"][0]
            data["camera2ego"].append(camera2ego)
            
            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            data["camera2lidar"].append(camera2lidar)
        
    return data, imgs


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
        if k == 'points':
            sample[k] = [torch.FloatTensor(example[k].data).cuda()]
        else:
            try:
                sample[k] = torch.unsqueeze(example[k].data, dim=0).cuda()
            except Exception as e:
               sample[k] = [example[k].data]
    
    return sample


class BevFusion:
    def __init__(
        self, cfg, box_type='LiDAR', ckpt=None, fuse_conv=True, calibrated_sensor_cfg=None):
        configs.load(cfg, recursive=True)
        self.cfg = Config(recursive_eval(configs), filename=cfg)
        self.calibrated_sensors = read_json(calibrated_sensor_cfg)
        self.camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        self.all_cam_intrinsics = {}
        self.cam_translations = {}
        self.cam_rotations = {}
        for cam in self.camera_types:
            self.all_cam_intrinsics[cam] = self.calibrated_sensors[cam]['camera_intrinsic']
            self.cam_translations[cam] = self.calibrated_sensors[cam]['translation']
            self.cam_rotations[cam] = self.calibrated_sensors[cam]['rotation']

        self.lidar2ego_translation = self.calibrated_sensors['LIDAR_TOP']['translation']
        self.lidar2ego_rotation = self.calibrated_sensors['LIDAR_TOP']['rotation']

        self.pipeline = Compose(self.cfg.data.test.pipeline)
        self.box_type_3d, self.box_mode_3d = get_box_type('LiDAR')
        self.model = build_model(self.cfg.model, test_cfg=self.cfg.get("test_cfg"))
        fp16_cfg = self.cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
        checkpoint = load_checkpoint(self.model, ckpt, map_location="cpu")
        if fuse_conv:
            self.model = fuse_conv_bn(self.model)
        
        if "CLASSES" in checkpoint.get("meta", {}):
            self.model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            self.model.CLASSES = self.cfg.object_classes
        self.model.cuda()
        self.model.eval()
    
    def __call__(self, data, bbox_score=0.35, out_dir='./viz_infer'):
        images, lidar_path, ego2global_translation, ego2global_rotation = data
        info = get_data(
            images, lidar_path, self.all_cam_intrinsics, self.lidar2ego_translation, self.lidar2ego_rotation, \
            ego2global_translation, ego2global_rotation, self.cam_translations, self.cam_rotations)
        info, imgs = build_data(info)
        sample = prepare_sample(info, self.pipeline, self.box_mode_3d, self.box_type_3d)

        # inference
        with torch.no_grad():
            outputs = self.model(**sample)

        # visualize
        bboxes = outputs[0]["boxes_3d"].tensor.numpy()
        scores = outputs[0]["scores_3d"].numpy()
        labels = outputs[0]["labels_3d"].numpy()

        if bbox_score is not None:
            indices = scores >= bbox_score
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]
        bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        metas = sample["metas"][0]
        name = "{}-{}".format(str(time.time()), str(1))
        output_images = {}
        j = 0
        import cv2
        if not len(bboxes):
            return None
        if "img" in sample:
            for k, img in imgs.items():
                image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cam_img = visualize_camera(
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][j],
                    classes=self.cfg.object_classes,
                )
                j += 1
                output_images[k] = cam_img
        return output_images
