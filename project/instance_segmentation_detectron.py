import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.export.torchscript import export_torchscript_with_instances
from detectron2.layers import paste_masks_in_image
from detectron2.modeling import build_model
from detectron2.structures import Boxes, BoxMode
from torch import Tensor, nn


def get_coco_dataset(annotation_path):
    annotation_path = Path(annotation_path)
    annotations = json.load(open(annotation_path, "r"))

    label_map = [c["name"] for c in annotations["categories"]]

    records = {
        image["id"]: {
            "file_name": str((annotation_path.parent / image["file_name"]).absolute()),
            "image_id": image["id"],
            "height": image["height"],
            "width": image["width"],
            "annotations": [],
        }
        for image in annotations["images"]
    }

    for annotation in annotations["annotations"]:
        records[annotation["image_id"]]["annotations"].append(
            {
                "bbox": annotation["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [annotation["segmentation"]],
                "category_id": annotation["category_id"],
                "iscrowd": annotation["iscrowd"],
            }
        )

    return label_map, [r for r in records.values() if len(r["annotations"]) > 0]


if __name__ == "__main__":
    labels, train_dataset = get_coco_dataset("data/train.json")
    _, test_dateset = get_coco_dataset("data/test.json")

    random.seed(1337)

    print(f"Dataset length: {len(train_dataset)}")
    DatasetCatalog.register("_train", lambda: train_dataset)
    DatasetCatalog.register("_test", lambda: test_dateset)

    MetadataCatalog.get("_train").set(thing_classes=labels)
    MetadataCatalog.get("_test").set(thing_classes=labels)

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("_train",)
    cfg.DATASETS.TEST = ("_test",)
    cfg.DATALOADER.NUM_WORKERS = 10
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = 300
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 500

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)

    log_dir = Path("./logs")
    last_version = len(list(log_dir.glob("version_*")))
    output_dir = log_dir / f"version_{last_version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(output_dir)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Let's save the configuration
    cfg.MODEL.WEIGHTS = str(output_dir / "model_final.pth")
    with open(output_dir / "config.yaml", "w") as f:
        f.write(cfg.dump())

    model = build_model(cfg)

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # Export to torchscript so we can run the net without detectron2
    # Taken from https://github.com/facebookresearch/detectron2/blob/v0.6/tools/deploy/export_model.py
    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = model
            self.eval()

    class ScriptableAdapter(ScriptableAdapterBase):
        def forward(self, image: Tensor) -> List[Dict[str, Tensor]]:
            image = image.permute(2, 0, 1)
            instances = self.model.inference(
                [{"image": image.float()}], do_postprocess=False
            )
            for instance in instances:
                # Resize the masks
                bitmasks = paste_masks_in_image(
                    instance.pred_masks[:, 0, :, :],
                    instance.pred_boxes,
                    (instance.image_size[0], instance.image_size[1]),
                    threshold=0.5,
                )
                instance.pred_masks = torch.as_tensor(bitmasks, dtype=torch.bool)
            return [i.get_fields() for i in instances]

    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": Tensor,
        "pred_keypoint_heatmaps": Tensor,
    }
    script_model = export_torchscript_with_instances(ScriptableAdapter(), fields)
    torch.jit.save(script_model, str(Path(cfg.MODEL.WEIGHTS).with_suffix(".pt")))
