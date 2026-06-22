import torch
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(
        self,
        S=7,
        B=2,
        C=20,
        lambda_coord=5,
        lambda_noobj=0.5,
        debug=False,
        debug_interval=100,
    ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.debug = debug
        self.debug_interval = max(1, debug_interval)
        self.forward_count = 0
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):
        """
        pred:   [N, S, S, C + B * 5]
                classes, then B boxes as [x_cell, y_cell, sqrt(w), sqrt(h), conf]
        target: [N, S, S, C + 5]
                classes, objectness, then [x_cell, y_cell, w, h]
        """
        N = pred.shape[0]
        obj_mask = target[..., self.C].unsqueeze(-1)
        obj_mask_s = obj_mask.squeeze(-1)

        class_loss = self.mse(
            obj_mask * pred[..., :self.C],
            obj_mask * target[..., :self.C],
        )

        pred_boxes = pred[..., self.C:self.C + self.B * 5].reshape(
            N, self.S, self.S, self.B, 5
        )
        pred_box1 = pred_boxes[..., 0, :]
        pred_box2 = pred_boxes[..., 1, :]
        target_box = target[..., self.C + 1:self.C + 5]

        device = pred.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.S, device=device),
            torch.arange(self.S, device=device),
            indexing="ij",
        )
        grid_x = grid_x.view(1, self.S, self.S)
        grid_y = grid_y.view(1, self.S, self.S)

        def cell_to_image(box, pred_wh_is_sqrt=False):
            x = (box[..., 0] + grid_x) / self.S
            y = (box[..., 1] + grid_y) / self.S
            if pred_wh_is_sqrt:
                sqrt_w = box[..., 2].clamp(min=-2.0, max=2.0)
                sqrt_h = box[..., 3].clamp(min=-2.0, max=2.0)
                w = sqrt_w.pow(2)
                h = sqrt_h.pow(2)
            else:
                w = box[..., 2]
                h = box[..., 3]
            return torch.stack([x, y, w, h], dim=-1)

        pred_box1_img = cell_to_image(pred_box1[..., :4], pred_wh_is_sqrt=True)
        pred_box2_img = cell_to_image(pred_box2[..., :4], pred_wh_is_sqrt=True)
        target_box_img = cell_to_image(target_box)

        iou1 = self.iou(pred_box1_img, target_box_img).detach()
        iou2 = self.iou(pred_box2_img, target_box_img).detach()

        best_box = (iou2 > iou1).float().unsqueeze(-1)
        best_box_s = best_box.squeeze(-1)
        best_pred_box = best_box * pred_box2 + (1 - best_box) * pred_box1
        best_iou = best_box_s * iou2 + (1 - best_box_s) * iou1

        coord_xy_loss = self.mse(
            obj_mask * best_pred_box[..., :2],
            obj_mask * target_box[..., :2],
        )
        coord_wh_loss = self.mse(
            obj_mask * best_pred_box[..., 2:4],
            obj_mask * torch.sqrt(target_box[..., 2:4].clamp(min=1e-6)),
        )
        coord_loss = coord_xy_loss + coord_wh_loss

        conf_pred1 = pred_box1[..., 4]
        conf_pred2 = pred_box2[..., 4]
        best_conf_pred = best_box_s * conf_pred2 + (1 - best_box_s) * conf_pred1

        conf_loss_obj = self.mse(
            obj_mask_s * best_conf_pred,
            obj_mask_s * best_iou,
        )

        noobj_mask_box1 = (1 - obj_mask_s) + obj_mask_s * best_box_s
        noobj_mask_box2 = (1 - obj_mask_s) + obj_mask_s * (1 - best_box_s)
        conf_loss_noobj = self.mse(
            noobj_mask_box1 * conf_pred1,
            torch.zeros_like(conf_pred1),
        ) + self.mse(
            noobj_mask_box2 * conf_pred2,
            torch.zeros_like(conf_pred2),
        )

        total_loss = (
            self.lambda_coord * coord_loss
            + conf_loss_obj
            + self.lambda_noobj * conf_loss_noobj
            + class_loss
        )
        self.forward_count += 1
        if self.debug and self.forward_count % self.debug_interval == 0:
            self.print_debug(
                total_loss=total_loss,
                coord_xy_loss=coord_xy_loss,
                coord_wh_loss=coord_wh_loss,
                conf_loss_obj=conf_loss_obj,
                conf_loss_noobj=conf_loss_noobj,
                class_loss=class_loss,
                obj_mask_s=obj_mask_s,
                best_box_s=best_box_s,
                best_iou=best_iou,
                best_conf_pred=best_conf_pred,
                pred_box1=pred_box1,
                pred_box2=pred_box2,
                target_box=target_box,
                pred_box1_img=pred_box1_img,
                pred_box2_img=pred_box2_img,
            )
        return total_loss / N

    def print_debug(
        self,
        total_loss,
        coord_xy_loss,
        coord_wh_loss,
        conf_loss_obj,
        conf_loss_noobj,
        class_loss,
        obj_mask_s,
        best_box_s,
        best_iou,
        best_conf_pred,
        pred_box1,
        pred_box2,
        target_box,
        pred_box1_img,
        pred_box2_img,
    ):
        with torch.no_grad():
            obj_count = int(obj_mask_s.sum().item())
            obj_bool = obj_mask_s.bool()
            noobj_count = int((1 - obj_mask_s).sum().item())

            if obj_count > 0:
                obj_iou = best_iou[obj_bool]
                obj_conf = best_conf_pred[obj_bool]
                target_xy = target_box[..., :2][obj_bool]
                target_wh = target_box[..., 2:4][obj_bool]
                pred_xy = (
                    best_box_s.unsqueeze(-1) * pred_box2[..., :2]
                    + (1 - best_box_s.unsqueeze(-1)) * pred_box1[..., :2]
                )[obj_bool]
                pred_sqrt_wh = (
                    best_box_s.unsqueeze(-1) * pred_box2[..., 2:4]
                    + (1 - best_box_s.unsqueeze(-1)) * pred_box1[..., 2:4]
                )[obj_bool]
                pred_wh = pred_sqrt_wh.clamp(min=-2.0, max=2.0).pow(2)
                iou_mean = obj_iou.mean().item()
                iou_max = obj_iou.max().item()
                conf_mean = obj_conf.mean().item()
                target_xy_min = target_xy.min().item()
                target_xy_max = target_xy.max().item()
                pred_xy_min = pred_xy.min().item()
                pred_xy_max = pred_xy.max().item()
                target_wh_min = target_wh.min().item()
                target_wh_max = target_wh.max().item()
                pred_wh_min = pred_wh.min().item()
                pred_wh_max = pred_wh.max().item()
                box2_ratio = best_box_s[obj_bool].mean().item()
            else:
                iou_mean = iou_max = conf_mean = 0.0
                target_xy_min = target_xy_max = 0.0
                pred_xy_min = pred_xy_max = 0.0
                target_wh_min = target_wh_max = 0.0
                pred_wh_min = pred_wh_max = 0.0
                box2_ratio = 0.0

            conf_all = torch.cat([pred_box1[..., 4].reshape(-1), pred_box2[..., 4].reshape(-1)])
            pred_img = torch.cat([
                pred_box1_img.reshape(-1, 4),
                pred_box2_img.reshape(-1, 4),
            ])
            nan_count = (
                torch.isnan(pred_box1).sum()
                + torch.isnan(pred_box2).sum()
                + torch.isnan(target_box).sum()
            ).item()
            inf_count = (
                torch.isinf(pred_box1).sum()
                + torch.isinf(pred_box2).sum()
                + torch.isinf(target_box).sum()
            ).item()

            print(
                "\n[YOLOLoss debug] "
                f"step={self.forward_count} obj={obj_count} noobj_cells={noobj_count} "
                f"total/N={(total_loss / max(1, obj_mask_s.shape[0])).item():.4f}"
            )
            print(
                "  loss raw: "
                f"xy={coord_xy_loss.item():.3f} wh={coord_wh_loss.item():.3f} "
                f"coord*w={(self.lambda_coord * (coord_xy_loss + coord_wh_loss)).item():.3f} "
                f"conf_obj={conf_loss_obj.item():.3f} "
                f"conf_noobj={conf_loss_noobj.item():.3f} "
                f"conf_noobj*w={(self.lambda_noobj * conf_loss_noobj).item():.3f} "
                f"class={class_loss.item():.3f}"
            )
            print(
                "  obj stats: "
                f"best_iou_mean={iou_mean:.4f} best_iou_max={iou_max:.4f} "
                f"best_conf_mean={conf_mean:.4f} box2_ratio={box2_ratio:.3f}"
            )
            print(
                "  xy range: "
                f"target=[{target_xy_min:.3f},{target_xy_max:.3f}] "
                f"pred=[{pred_xy_min:.3f},{pred_xy_max:.3f}]"
            )
            print(
                "  wh range(image ratio): "
                f"target=[{target_wh_min:.3f},{target_wh_max:.3f}] "
                f"pred=[{pred_wh_min:.3f},{pred_wh_max:.3f}]"
            )
            print(
                "  all pred: "
                f"conf=[{conf_all.min().item():.3f},{conf_all.max().item():.3f}] "
                f"img_xywh=[{pred_img.min().item():.3f},{pred_img.max().item():.3f}] "
                f"nan={int(nan_count)} inf={int(inf_count)}"
            )

    @staticmethod
    def iou(box1, box2):
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

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        box1_area = (box1_x2 - box1_x1).clamp(min=0) * (box1_y2 - box1_y1).clamp(min=0)
        box2_area = (box2_x2 - box2_x1).clamp(min=0) * (box2_y2 - box2_y1).clamp(min=0)
        union = box1_area + box2_area - inter_area + 1e-6
        return inter_area / union
