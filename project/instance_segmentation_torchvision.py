from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from fastcore.basics import first
from icevision import tfms
from icevision.data.data_splitter import SingleSplitSplitter
from icevision.data.dataset import Dataset
from icevision.metrics.coco_metric import COCOMetric, COCOMetricType
from icevision.models.torchvision import mask_rcnn
from icevision.parsers import COCOMaskParser
from icevision.visualize.show_data import show_preds, show_samples
from torch.optim import Adam


def train():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    train_tfms = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()]
    )
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(384), tfms.A.Normalize()])

    train_ps = COCOMaskParser("data/train.json", "data")
    test_ps = COCOMaskParser("data/test.json", "data")

    train_ds = Dataset(
        train_ps.parse(data_splitter=SingleSplitSplitter())[0], train_tfms
    )
    valid_ds = Dataset(
        test_ps.parse(data_splitter=SingleSplitSplitter())[0], valid_tfms
    )

    samples = [train_ds[0]]
    show_samples(samples, ncols=1)

    train_dl = mask_rcnn.train_dl(train_ds, batch_size=1, num_workers=16, shuffle=True)
    valid_dl = mask_rcnn.valid_dl(valid_ds, batch_size=1, num_workers=16, shuffle=False)

    mask_rcnn.show_batch(first(valid_dl), ncols=1)

    model = mask_rcnn.model(
        backbone=mask_rcnn.backbones.resnet18_fpn(),
        num_classes=train_ps.class_map.num_classes,
    )

    class LightModel(mask_rcnn.lightning.ModelAdapter):
        def configure_optimizers(self):
            return Adam(self.parameters(), lr=5e-4)

    light_model = LightModel(
        model, metrics=[COCOMetric(metric_type=COCOMetricType.mask)]
    )

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(light_model, train_dl, valid_dl)

    infer_dl = mask_rcnn.infer_dl(valid_ds, batch_size=1, shuffle=False)
    preds = mask_rcnn.predict_from_dl(model, infer_dl, keep_images=True)

    show_preds(preds=preds, ncols=1)

    input_sample = torch.randn((3, 224, 224))

    dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
    torch.onnx.export(
        model,
        str(Path(trainer.checkpoint_callback.best_model_path).with_suffix(".onnx")),
        input_sample,
        export_params=True,
        opset_version=13,
        input_names=["images"],
        output_names=["pred_boxes", "scores", "pred_classes", "pred_masks"],
        dynamic_axes=dynamic_axes,
    )


if __name__ == "__main__":
    train()
