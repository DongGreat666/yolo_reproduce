# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/13 12:39

import torch
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()

        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss(reduction="sum")

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, pred, target):
        """
        pred:   [B, S, S, C + B*5] = [B,7,7,30]
        target: [B, S, S, C + 5]   = [B,7,7,25]
        """

        N = pred.shape[0]

        # =========================
        # 1. object mask
        # =========================
        obj_mask = target[..., self.C].unsqueeze(-1)   # [B,S,S,1]

        # =========================
        # 2. class loss
        # =========================
        class_loss = self.mse(
            obj_mask * pred[..., :self.C],
            obj_mask * target[..., :self.C]
        )

        # =========================
        # 3. split boxes
        # =========================
        pred_boxes = pred[..., self.C:self.C + self.B * 5]
        pred_boxes = pred_boxes.reshape(N, self.S, self.S, self.B, 5)

        pred_box1 = pred_boxes[..., 0, :]
        pred_box2 = pred_boxes[..., 1, :]

        target_box = target[..., self.C + 1:self.C + 5]   # [B,S,S,4]

        # =========================
        # 4. IoU
        # =========================
        iou1 = self.iou(pred_box1[..., :4], target_box)
        iou2 = self.iou(pred_box2[..., :4], target_box)

        # =========================
        # 5. responsible box selection（关键：不用 gather）
        # =========================
        best_box = (iou2 > iou1).float().unsqueeze(-1)

        best_pred_box = best_box * pred_box2 + (1 - best_box) * pred_box1

        # =========================
        # 6. coord loss
        # =========================
        coord_loss = self.mse(
            obj_mask * best_pred_box[..., :2],
            obj_mask * target_box[..., :2]
        ) + self.mse(
            obj_mask * torch.sqrt(best_pred_box[..., 2:4].clamp(1e-6)),
            obj_mask * torch.sqrt(target_box[..., 2:4].clamp(1e-6))
        )

        # =========================
        # 7. confidence loss
        # =========================
        conf_pred1 = pred_box1[..., 4]
        conf_pred2 = pred_box2[..., 4]

        conf_target = obj_mask.squeeze(-1)

        conf_loss_obj = self.mse(
            obj_mask.squeeze(-1) * (best_box.squeeze(-1) * conf_pred2 + (1 - best_box.squeeze(-1)) * conf_pred1),
            conf_target
        )

        conf_loss_noobj = self.mse(
            (1 - obj_mask).squeeze(-1) * conf_pred1,
            torch.zeros_like(conf_pred1)
        ) + self.mse(
            (1 - obj_mask).squeeze(-1) * conf_pred2,
            torch.zeros_like(conf_pred2)
        )

        # =========================
        # 8. total loss
        # =========================
        loss = (
            self.lambda_coord * coord_loss +
            class_loss +
            conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj
        )

        return loss / N

    def iou(self, box1, box2):
        """
        box: [x, y, w, h]
        """

        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2

        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2

        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        union = box1_area + box2_area - inter_area + 1e-6

        return inter_area / union