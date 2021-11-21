from argparse import ArgumentParser

import cv2
import torch
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer


def test(model_path, images):
    model = torch.jit.load(model_path)

    for image in images:
        image = cv2.imread(image)

        # TODO: Hardcoded resize, make it a bit smarter
        resized = cv2.resize(image, (1067, 800))

        outputs = model(torch.as_tensor(resized))
        outputs = {k: v.cpu().detach().numpy() for k, v in outputs[0].items()}

        v = Visualizer(resized[:, :, ::-1], instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(Instances(resized.shape[:2], **outputs))
        img = v.get_image()[:, :, ::-1]
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("images", metavar="N", type=str, nargs="+", help="images")
    parser.add_argument("--model_path", default="", type=str)

    args = parser.parse_args()

    test(args.model_path, args.images)
