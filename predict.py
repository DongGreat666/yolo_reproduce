import os
import random

import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

import config
from model.factory import build_yolo_model
from utils.dataset import VOCDataset, VOC_CLASSES
from utils.checkpoint import load_model_state


def iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes, iou_threshold=0.5):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        rest = []
        for box in boxes:
            if box[0] != best[0] or iou_xyxy(best[2:], box[2:]) < iou_threshold:
                rest.append(box)
        boxes = rest

    return keep


def decode_prediction(pred, conf_threshold=0.05):
    """
    pred: [S, S, C + B * 5]
    return: [class_id, score, x1, y1, x2, y2], normalized to image size.
    """
    boxes = []
    S = config.S
    B = config.B
    C = config.NUM_CLASSES

    pred = pred.detach().cpu()
    class_scores = pred[..., :C]
    pred_boxes = pred[..., C:C + B * 5].reshape(S, S, B, 5)

    for i in range(S):
        for j in range(S):
            for b in range(B):
                x_cell, y_cell, sqrt_w, sqrt_h, conf = pred_boxes[i, j, b]
                x = (x_cell.item() + j) / S
                y = (y_cell.item() + i) / S
                w = sqrt_w.clamp(min=-2.0, max=2.0).item() ** 2
                h = sqrt_h.clamp(min=-2.0, max=2.0).item() ** 2
                conf = conf.item()

                for class_id in range(C):
                    score = conf * class_scores[i, j, class_id].item()
                    if score < conf_threshold:
                        continue

                    x1 = max(0, min(1, x - w / 2))
                    y1 = max(0, min(1, y - h / 2))
                    x2 = max(0, min(1, x + w / 2))
                    y2 = max(0, min(1, y + h / 2))
                    boxes.append([class_id, score, x1, y1, x2, y2])

    return nms(boxes, iou_threshold=0.5)


def draw_boxes(image, boxes, save_path):
    draw = ImageDraw.Draw(image)
    w_img, h_img = image.size

    for class_id, score, x1, y1, x2, y2 in boxes:
        x1 = int(x1 * w_img)
        y1 = int(y1 * h_img)
        x2 = int(x2 * w_img)
        y2 = int(y2 * h_img)
        label = f"{VOC_CLASSES[class_id]} {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], width=3, outline='black')
        draw.text((x1, max(0, y1 - 12)), label, fill='black')

    image.save(save_path)


def load_model(device):
    model = build_yolo_model(config).to(device)

    ckpt_path = config.BEST_MODEL_PATH
    if not os.path.exists(ckpt_path):
        ckpt_path = config.LAST_MODEL_PATH

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        load_model_state(model, checkpoint["model_state_dict"])
    else:
        load_model_state(model, checkpoint)

    model.eval()
    print("Loaded checkpoint:", ckpt_path)
    return model


def main():
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    device = torch.device(config.DEVICE)
    print("Using Device:", device)

    model = load_model(device)
    dataset = VOCDataset(
        root_dir=config.DATA_DIR,
        year="2007",
        image_set="test",
        S=config.S,
        B=config.B,
        C=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
        train=False,
    )

    indices = random.sample(range(len(dataset)), k=5)
    for idx, data_idx in enumerate(indices):
        image_id = dataset.ids[data_idx]
        image_path = os.path.join(dataset.image_dir, image_id + ".jpg")
        image = Image.open(image_path).convert("RGB")
        input_img = TF.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        input_tensor = TF.to_tensor(input_img).unsqueeze(0).to(device)
        if config.NORMALIZE_IMAGES:
            input_tensor = TF.normalize(
                input_tensor.squeeze(0),
                mean=config.IMAGE_MEAN,
                std=config.IMAGE_STD,
            ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_tensor)[0]

        boxes = decode_prediction(pred, conf_threshold=0.25)
        save_path = os.path.join(config.OUTPUTS_DIR, f"pred_{idx}_{image_id}.jpg")
        draw_boxes(image.copy(), boxes, save_path)
        print(f"{image_id}: {len(boxes)} boxes -> {save_path}")


if __name__ == "__main__":
    main()
