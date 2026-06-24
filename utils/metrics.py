import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
from tqdm import tqdm

from utils.decode import box_iou, decode_predictions


def parse_voc_ground_truth(dataset):
    # 解析 PASCAL VOC 数据集的 XML 标签文件，
    # 提取出所有图片的真实框，并按照类别对它们重新组织

    ground_truths = defaultdict(dict)  # 创建一套嵌套的默认字典，最终结构：{类别ID: {图片ID: [真值框1, 真值框2, ...]}}
                                       # 用 defaultdict 可以防止后面往不存在的键里写数据时报 KeyError
    class_to_idx = dataset.class_to_idx  # 从数据集对象中获取字符串类名到数字 ID 的映射字典

    for image_id in dataset.ids:
        anno_path = os.path.join(dataset.anno_dir, image_id + ".xml")
        tree = ET.parse(anno_path)  # 使用 Python 内置的 xml.etree.ElementTree 库来读取并解析对应的 .xml 标签文件
        root = tree.getroot()
        size = root.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)

        for obj in root.findall("object"):  # 一幅图里有多个物体，逐个抠出
            class_name = obj.find("name").text
            class_id = class_to_idx[class_name]
            difficult_node = obj.find("difficult")  # 困难样本
            # 如果物体遮挡严重或太小，会被标为 1（困难样本）。
            # 在标准 mAP 计算中，通常会忽略这些困难样本，不让它们拉低模型的评估分数。
            difficult = difficult_node is not None and difficult_node.text == "1"
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text) / img_w
            ymin = float(bndbox.find("ymin").text) / img_h
            xmax = float(bndbox.find("xmax").text) / img_w
            ymax = float(bndbox.find("ymax").text) / img_h

            # 打包
            ground_truths[class_id].setdefault(image_id, []).append(
                {
                    "box": [
                        max(0.0, min(1.0, xmin)),
                        max(0.0, min(1.0, ymin)),
                        max(0.0, min(1.0, xmax)),
                        max(0.0, min(1.0, ymax)),
                    ],
                    "difficult": difficult,
                    "matched": False,
                }
            )

    return ground_truths


def average_precision(detections, ground_truths, num_gt, iou_threshold=0.5, ignore_difficult=True):
    if num_gt == 0:
        return None
    if not detections:
        return 0.0

    # 1. 必须按模型预测的置信度 score 降序排列！
    # 目标检测的博弈逻辑：模型对自己最自信的框（score最高的），必须最先被拉去算对错！
    detections = sorted(detections, key=lambda item: item["score"], reverse=True)
    # 2. 建立两个全 0 张量，长度等于预测框总数。用来记录每一个框究竟是 True Positive 还是 False Positive
    tp = torch.zeros(len(detections))
    fp = torch.zeros(len(detections))

    for det_idx, detection in enumerate(detections):
        # 拿当前预测框去真值账本里，找出同一张图片上的同类真值框
        image_gts = ground_truths.get(detection["image_id"], [])
        best_iou = 0.0
        best_gt_idx = -1
        # 【打擂台】：把当前预测框和这张图里所有的真值框挨个算 IoU，找出亲缘关系最近（IoU最大）的那个真值框
        for gt_idx, gt in enumerate(image_gts):
            iou = box_iou(detection["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # 如果最大的 IoU 跨过了及格线（比如 0.5），说明定位大致对了
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            best_gt = image_gts[best_gt_idx]
            # 细节：如果这个匹配上的真值是困难样本，VOC规则选择“原地忽视”，不算错也不算对
            if ignore_difficult and best_gt["difficult"]:
                # VOC rule: a detection matched to a difficult object is ignored.
                continue
            # 【反刷分作弊铁律】：
            # 如果这个真值框之前【没有】被别的预测框匹配过 (!matched)
            if not best_gt["matched"]:
                tp[det_idx] = 1  # 恭喜你，这是一个名正言顺的 True Positive！
                best_gt["matched"] = True  # 【立刻锁死】这个真值框已经被名花有主了！
            else:
                # 如果这个真值框已经被之前置信度更高的框捷足先登了
                # 你虽然定位也准，但对不起，你来晚了，只能被判定为“重复预测”的 False Positive！
                fp[det_idx] = 1
        else:
            # 如果 IoU 连及格线都没到，说明纯属瞎猜，直接判定为垃圾背景框（False Positive）
            fp[det_idx] = 1

    # 1. 累加求和：计算出随着置信度往下走，当前累计抓对了几个（tp_cumsum）和累计抓错了几个（fp_cumsum）
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    # 2. 根据公式算出每个阶段对应的召回率（Recall）和精确率（Precision）
    # Recall = 抓对的数量 / 验证集已知的真实总数
    # Precision = 抓对的数量 / 丢出来的预测总数
    recalls = tp_cumsum / (num_gt + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # 3. 边界修补：为了闭合 PR 曲线面积，在两头强行补齐 [0.0] 和 [1.0]
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    # 4. 【灵魂单调化平滑（VOC 11点/全积分标准）】：
    # 从右往左遍历，强制让 Precision 变成单调递减（即去除 PR 曲线上的局部锯齿形下凹抖动）
    # 公式：当前位置的 Precision 必须等于 自身与右边所有元素最大值 之间的较大者
    for idx in range(precisions.numel() - 1, 0, -1):
        precisions[idx - 1] = torch.maximum(precisions[idx - 1], precisions[idx])

    # 5. 寻找 Recall 的跳变点（即 PR 曲线产生台阶的地方）
    changing_points = torch.where(recalls[1:] != recalls[:-1])[0]

    # 6. 计算微元面积：用（右边Recall - 左边Recall）× 当前台阶的平滑后Precision
    # 这就是微积分里标准的矩形面积累加，最终得到 PR 曲线下的总面积，即 Average Precision (AP)！
    ap = torch.sum(
        (recalls[changing_points + 1] - recalls[changing_points])
        * precisions[changing_points + 1]
    )
    return ap.item()


def evaluate_map(model, dataset, device, iou_threshold=0.5,
                 conf_threshold=0.0, nms_threshold=0.5, ignore_difficult=True):
    model.eval()
    # 1. 把全验证集的真值按 {类: {图: [框]}} 格式加载进来
    ground_truths = parse_voc_ground_truth(dataset)
    detections_by_class = defaultdict(list)  # 准备一个列表，按类别分类全验证集所有的预测框

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="mAP"):  # tqdm可以显示进度
            image, _ = dataset[idx]
            image_id = dataset.ids[idx]
            # 2. 图片入网前哨站：升维成 4D 并送入 GPU 推理
            pred = model(image.unsqueeze(0).to(device))[0]
            # 3. 解码并做 NMS 过滤，拿到这张图上所有的胜出框
            detections = decode_predictions(pred, S=dataset.S, B=dataset.B, C=dataset.C,
                                            conf_threshold=conf_threshold,nms_threshold=nms_threshold)
            # 4. 贴上图片身份证标签，并按类别分流归类
            for detection in detections:
                detection = dict(detection)
                detection["image_id"] = image_id
                detections_by_class[detection["class_id"]].append(detection)

    ap_by_class = {}
    # 5. 遍历每一个类别
    for class_id in range(dataset.C):
        class_gts = ground_truths.get(class_id, {})
        # 6. 【核心细节】：统计该类别在整个验证集里的“有效真值总数 (num_gt)”
        # 必须排除掉那些高难度的 difficult 样本，不让它们参与分母计算
        num_gt = sum(
            1
            for items in class_gts.values()
            for item in items
            if not (ignore_difficult and item["difficult"])
        )
        # 7. 把这个科目所有的预测框和真值丢给核心函数算 AP
        ap = average_precision(
            detections_by_class.get(class_id, []),
            class_gts,
            num_gt,
            iou_threshold=iou_threshold,
            ignore_difficult=ignore_difficult,
        )
        if ap is not None:
            ap_by_class[class_id] = ap
    # 8. 将所有科目的 AP 加起来，除以科目总数，得到期末总评成绩：mean AP (mAP)
    mean_ap = sum(ap_by_class.values()) / max(1, len(ap_by_class))
    return mean_ap, ap_by_class
