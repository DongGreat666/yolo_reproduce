from model.yolov1 import YOLOv1
from model.resnet_yolo import ResNetYOLO


def build_yolo_model(config):
    if config.MODEL_NAME == "yolov1":
        return YOLOv1(
            in_channels=3,
            split_size=config.S,
            num_boxes=config.B,
            num_classes=config.NUM_CLASSES,
            detection_head=config.DETECTION_HEAD,
        )

    if config.MODEL_NAME == "resnet_yolo":
        return ResNetYOLO(
            split_size=config.S,
            num_boxes=config.B,
            num_classes=config.NUM_CLASSES,
            pretrained=config.RESNET_PRETRAINED,
            detection_head=config.DETECTION_HEAD,
        )

    raise ValueError(f"Unknown MODEL_NAME: {config.MODEL_NAME}")
