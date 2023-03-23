import argparse
import os
import time
import warnings

import mmcv
import onnx
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from onnxsim import simplify
from tqdm import tqdm

from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
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
    args, opts = parser.parse_known_args()
    return args, opts


def main():
    args, opts = parse_args()

    # cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    model.eval()
    with torch.no_grad():
        for data in data_loader:

            img = data["img"].data[0]
            points = data["points"].data[0][0]
            camera2ego = data["camera2ego"].data[0]
            lidar2ego = data["lidar2ego"].data[0]
            lidar2camera = data["lidar2camera"].data[0]
            lidar2image = data["lidar2image"].data[0]
            camera_intrinsics = data["camera_intrinsics"].data[0]
            camera2lidar = data["camera2lidar"].data[0]
            img_aug_matrix = data["img_aug_matrix"].data[0]
            lidar_aug_matrix = data["lidar_aug_matrix"].data[0]
            metas = data["metas"].data
            x = (img, points, camera2ego, lidar2ego, lidar2camera, lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix, lidar_aug_matrix)
            lidar2img_meta = metas[0][0]["lidar2image"]

            for i, arr in enumerate(lidar2img_meta):
                metas[0][0]["lidar2image"][i] = torch.from_numpy(arr)
            
            for k, v in metas[0][0].items():
                print(k, type(v))
            xx
            # metas = data["metas"].data

            # from functools import partial

            # model.forward = partial(
            #     model.forward_test,
            #     metas=metas,
            #     rescale=True,
            # )

            torch.onnx.export(
                model,
                x,
                "model.onnx",
                input_names=["img", "points", "camera2ego", "lidar2camera", "lidar2ego", "camera_intrinsics", "camera2lidar"
                            "img_aug_matrix", "lidar_aug_matrix"],
                opset_version=13,
                do_constant_folding=True,
            )
            model = onnx.load("model.onnx")
            model, _ = simplify(model)
            onnx.save(model, "model.onnx")
            return


if __name__ == "__main__":
    main()
