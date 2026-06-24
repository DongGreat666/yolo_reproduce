# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/13 12:39

import torch

import config


def box_iou(box1, box2):
    # 求交集的位置
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 求交集面积
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    # 分别求两个框的面积
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    # 求交并比，1e-6 用于防止分母为 0
    return inter / (area1 + area2 - inter + 1e-6)


def nms(detections, iou_threshold=0.5):
    """Apply class-aware NMS to detection dictionaries."""
    # 对框按置信度从高到低排序
    detections = sorted(detections, key=lambda item: item["score"], reverse=True)
    keep = []

    while detections:
        # 每次保留当前分数最高的框
        best = detections.pop(0)
        keep.append(best)
        # 不同类别互不抑制；同类别且与最佳框 IoU 较大的重复框被删除
        detections = [
            detection
            for detection in detections
            if detection["class_id"] != best["class_id"]
            or box_iou(best["box"], detection["box"]) < iou_threshold
        ]

    return keep


def decode_predictions(pred, S, B, C, conf_threshold=0.0, nms_threshold=0.5, class_mode="all"):
    """
    Decode one YOLOv1 output into detection dictionaries.

    class_mode="all" keeps class-specific scores for AP evaluation.
    class_mode="best" keeps only the strongest class in each grid cell for display.
    """
    if class_mode not in {"all", "best"}:
        raise ValueError(f"Unknown class_mode: {class_mode}")

    # 解码预测值
    detections = []
    pred = pred.detach().cpu()
    # 前 C 个值是每个网格的类别分数
    class_scores = pred[..., :C]
    # 后 B*5 个值是 B 个预测框，每个框为 x、y、sqrt(w)、sqrt(h)、confidence
    pred_boxes = pred[..., C:C + B * 5].reshape(S, S, B, 5)

    # 遍历每一个网格
    for row in range(S):
        for col in range(S):
            if class_mode == "best":
                # 可视化时每一个网格只取分数最高的类别，避免一张图产生过多类别框
                class_ids = [int(torch.argmax(class_scores[row, col]).item())]
            else:
                # 计算 AP 时保留所有类别分数，供每个类别分别构建 PR 曲线
                class_ids = range(C)

            # 每个网格由 B 个候选框负责，依次解码
            for box_idx in range(B):
                # 第 box_idx 个框的五个原始预测值：中心偏移、宽高平方根和置信度
                x_cell, y_cell, sqrt_w, sqrt_h, conf = pred_boxes[row, col, box_idx]
                # 将网格内相对中心坐标还原到整张图的 0~1 相对坐标：
                # 加上所在列/行，再除以总网格数 S
                x = (x_cell.item() + col) / S
                y = (y_cell.item() + row) / S
                # 训练目标使用 sqrt(w)、sqrt(h)，解码时平方还原为图像宽高比例
                # clamp 用于限制异常预测，避免平方后产生过大的框
                w = sqrt_w.clamp(min=-2.0, max=2.0).item() ** 2
                h = sqrt_h.clamp(min=-2.0, max=2.0).item() ** 2

                # 将中心点与宽高转换成左上角、右下角坐标，并裁剪到图像范围
                x1 = max(0.0, min(1.0, x - w / 2))
                y1 = max(0.0, min(1.0, y - h / 2))
                x2 = max(0.0, min(1.0, x + w / 2))
                y2 = max(0.0, min(1.0, y + h / 2))
                # 裁剪后没有有效面积的框直接丢弃
                if x2 <= x1 or y2 <= y1:
                    continue

                for class_id in class_ids:
                    # YOLOv1 类别检测分数 = 目标置信度 × 条件类别分数
                    score = conf.item() * class_scores[row, col, class_id].item()
                    # 只保留高于阈值的检测框
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


def decode_prediction(pred, conf_threshold=0.1, nms_threshold=0.5):
    """供 predict.py 使用，返回画框函数需要的列表格式。"""
    # 实际坐标解码和 NMS 复用上面的公共函数；这里只选择最佳类别并转换输出格式
    detections = decode_predictions(
        pred,
        S=config.S,
        B=config.B,
        C=config.NUM_CLASSES,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        class_mode="best",
    )
    return [
        [detection["class_id"], detection["score"], *detection["box"]]
        for detection in detections
    ]
