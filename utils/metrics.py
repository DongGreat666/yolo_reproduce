import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
from tqdm import tqdm


def xywh_to_xyxy(box):
    _, x, y, w, h = box
    return [
        max(0.0, x - w / 2),
        max(0.0, y - h / 2),
        min(1.0, x + w / 2),
        min(1.0, y + h / 2),
    ]


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes, iou_threshold=0.5):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda item: item["score"], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        rest = []
        for box in boxes:
            if box["class_id"] != best["class_id"]:
                rest.append(box)
            elif box_iou(best["box"], box["box"]) < iou_threshold:
                rest.append(box)
        boxes = rest

    return keep


def decode_predictions(pred, S, B, C, conf_threshold=0.0, nms_threshold=0.5):
    """
    Decode one YOLOv1 prediction tensor into normalized xyxy detections.
    The detection score follows the paper: class-specific score = class score * confidence.
    """
    detections = []
    pred = pred.detach().cpu()
    class_scores = pred[..., :C]
    pred_boxes = pred[..., C:C + B * 5].reshape(S, S, B, 5)

    for row in range(S):
        for col in range(S):
            for box_idx in range(B):
                x_cell, y_cell, sqrt_w, sqrt_h, conf = pred_boxes[row, col, box_idx]
                x = (x_cell.item() + col) / S
                y = (y_cell.item() + row) / S
                w = sqrt_w.clamp(min=-2.0, max=2.0).item() ** 2
                h = sqrt_h.clamp(min=-2.0, max=2.0).item() ** 2
                conf = conf.item()

                x1 = max(0.0, min(1.0, x - w / 2))
                y1 = max(0.0, min(1.0, y - h / 2))
                x2 = max(0.0, min(1.0, x + w / 2))
                y2 = max(0.0, min(1.0, y + h / 2))
                if x2 <= x1 or y2 <= y1:
                    continue

                for class_id in range(C):
                    score = conf * class_scores[row, col, class_id].item()
                    if score < conf_threshold:
                        continue
                    detections.append(
                        {
                            "class_id": class_id,
                            "score": score,
                            "box": [x1, y1, x2, y2],
                        }
                    )

    return nms(detections, iou_threshold=nms_threshold)


def parse_voc_ground_truth(dataset):
    ground_truths = defaultdict(dict)
    class_to_idx = dataset.class_to_idx

    for image_id in dataset.ids:
        anno_path = os.path.join(dataset.anno_dir, image_id + ".xml")
        tree = ET.parse(anno_path)
        root = tree.getroot()
        size = root.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_id = class_to_idx[class_name]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text) / img_w
            ymin = float(bndbox.find("ymin").text) / img_h
            xmax = float(bndbox.find("xmax").text) / img_w
            ymax = float(bndbox.find("ymax").text) / img_h

            ground_truths[class_id].setdefault(image_id, []).append(
                {
                    "box": [
                        max(0.0, min(1.0, xmin)),
                        max(0.0, min(1.0, ymin)),
                        max(0.0, min(1.0, xmax)),
                        max(0.0, min(1.0, ymax)),
                    ],
                    "matched": False,
                }
            )

    return ground_truths


def average_precision(detections, ground_truths, num_gt, iou_threshold=0.5):
    if num_gt == 0:
        return None
    if not detections:
        return 0.0

    detections = sorted(detections, key=lambda item: item["score"], reverse=True)
    tp = torch.zeros(len(detections))
    fp = torch.zeros(len(detections))

    for det_idx, detection in enumerate(detections):
        image_gts = ground_truths.get(detection["image_id"], [])
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(image_gts):
            iou = box_iou(detection["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            if not image_gts[best_gt_idx]["matched"]:
                tp[det_idx] = 1
                image_gts[best_gt_idx]["matched"] = True
            else:
                fp[det_idx] = 1
        else:
            fp[det_idx] = 1

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = tp_cumsum / (num_gt + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    for idx in range(precisions.numel() - 1, 0, -1):
        precisions[idx - 1] = torch.maximum(precisions[idx - 1], precisions[idx])

    changing_points = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum(
        (recalls[changing_points + 1] - recalls[changing_points])
        * precisions[changing_points + 1]
    )
    return ap.item()


def evaluate_map(
    model,
    dataset,
    device,
    iou_threshold=0.5,
    conf_threshold=0.0,
    nms_threshold=0.5,
):
    model.eval()
    ground_truths = parse_voc_ground_truth(dataset)
    detections_by_class = defaultdict(list)

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="mAP"):
            image, _ = dataset[idx]
            image_id = dataset.ids[idx]
            pred = model(image.unsqueeze(0).to(device))[0]
            detections = decode_predictions(
                pred,
                S=dataset.S,
                B=dataset.B,
                C=dataset.C,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
            )

            for detection in detections:
                detection = dict(detection)
                detection["image_id"] = image_id
                detections_by_class[detection["class_id"]].append(detection)

    ap_by_class = {}
    for class_id in range(dataset.C):
        class_gts = ground_truths.get(class_id, {})
        num_gt = sum(len(items) for items in class_gts.values())
        ap = average_precision(
            detections_by_class.get(class_id, []),
            class_gts,
            num_gt,
            iou_threshold=iou_threshold,
        )
        if ap is not None:
            ap_by_class[class_id] = ap

    mean_ap = sum(ap_by_class.values()) / max(1, len(ap_by_class))
    return mean_ap, ap_by_class
