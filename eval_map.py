import os

import torch

import config
from model.factory import build_yolo_model
from utils.dataset import VOCDataset, VOC_CLASSES
from utils.metrics import evaluate_map
from utils.checkpoint import load_model_state


def load_model(device, checkpoint_path=None):
    model = build_yolo_model(config).to(device)

    if checkpoint_path is None:
        checkpoint_path = config.BEST_MODEL_PATH
        if not os.path.exists(checkpoint_path):
            checkpoint_path = config.LAST_MODEL_PATH

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        load_model_state(model, checkpoint["model_state_dict"])
    else:
        load_model_state(model, checkpoint)

    model.eval()
    print("Loaded checkpoint:", checkpoint_path)
    return model


def main():
    device = torch.device(config.DEVICE)
    print("Using Device:", device)

    dataset = VOCDataset(
        root_dir=config.DATA_DIR,
        year="2007",
        image_set="test",
        train=False,
        S=config.S,
        B=config.B,
        C=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
    )
    model = load_model(device)

    mean_ap, ap_by_class = evaluate_map(
        model,
        dataset,
        device,
        iou_threshold=config.MAP_IOU_THRESHOLD,
        conf_threshold=config.MAP_CONF_THRESHOLD,
        nms_threshold=config.MAP_NMS_THRESHOLD,
        ignore_difficult=config.MAP_IGNORE_DIFFICULT,
    )

    print("Ignore difficult:", config.MAP_IGNORE_DIFFICULT)
    print(f"mAP@{config.MAP_IOU_THRESHOLD:.2f}: {mean_ap:.4f}")
    for class_id, ap in sorted(ap_by_class.items()):
        print(f"{VOC_CLASSES[class_id]:>12s}: {ap:.4f}")


if __name__ == "__main__":
    main()
