import os

import torch


# =========================== Common ===========================
PROJECT_NAME = "YoLo"

# Choose one: "yolov1" or "resnet_yolo".
MODEL_NAME = "resnet_yolo"
# Choose one: "fc_head" or "cnn_head".
DETECTION_HEAD = "cnn_head"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_ROOT = os.path.join(BASE_DIR, "checkpoints")
OUTPUTS_ROOT = os.path.join(BASE_DIR, "outputs")
LOG_ROOT = os.path.join(BASE_DIR, "logs")

# =====================================首次是否要加载============================================
# 后面打开 RESUME=True
RESUME = False
BACKBONE_INIT_PATH = os.path.join(
    CHECKPOINT_ROOT, "yolov1", "best_yolov1.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_DATA_PARALLEL = True
GPU_IDS = [0, 1]

IMG_SIZE = 448
NUM_CLASSES = 20
S = 7
B = 2

BATCH_SIZE = 64
EPOCHS = 135
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
GRAD_CLIP_NORM = 10.0

WARMUP_EPOCHS = 5
LR_WARMUP_START = 1e-5
LR_WARMUP_END = 1e-4
LR_STAGE1_EPOCHS = 25
LR_STAGE2_EPOCHS = 50
LR_STAGE3_EPOCHS = 55

LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
LOSS_DEBUG = True
LOSS_DEBUG_INTERVAL = 100

MAP_EVAL_INTERVAL = 5
MAP_IOU_THRESHOLD = 0.5
MAP_NMS_THRESHOLD = 0.5
MAP_CONF_THRESHOLD = 0.0
# VOC official evaluation excludes difficult objects from the positive count and
# ignores detections matched to them.
MAP_IGNORE_DIFFICULT = True


# =========================== yolov1 ===========================
# Original YOLOv1-style Darknet backbone.
YOLOV1_PRETRAINED_BACKBONE_PATH = ""
YOLOV1_NORMALIZE_IMAGES = False
YOLOV1_IMAGE_MEAN = [0.0, 0.0, 0.0]
YOLOV1_IMAGE_STD = [1.0, 1.0, 1.0]


# =========================== resnet_yolo ===========================
# YOLOv1 head/loss with a torchvision ResNet34 backbone.
RESNET_YOLO_PRETRAINED = True
RESNET_YOLO_FREEZE_BACKBONE_EPOCHS = 10
RESNET_YOLO_UNFREEZE_LAYER4_EPOCH = 10
RESNET_YOLO_UNFREEZE_LAYER3_EPOCH = 40
RESNET_YOLO_NORMALIZE_IMAGES = True
RESNET_YOLO_IMAGE_MEAN = [0.485, 0.456, 0.406]
RESNET_YOLO_IMAGE_STD = [0.229, 0.224, 0.225]


# =========================== Active model ===========================
if MODEL_NAME == "yolov1":
    MODEL_BACKBONE = "yolov1"
    PRETRAINED_BACKBONE_PATH = YOLOV1_PRETRAINED_BACKBONE_PATH
    NORMALIZE_IMAGES = YOLOV1_NORMALIZE_IMAGES
    IMAGE_MEAN = YOLOV1_IMAGE_MEAN
    IMAGE_STD = YOLOV1_IMAGE_STD
elif MODEL_NAME == "resnet_yolo":
    MODEL_BACKBONE = "resnet_yolo"
    RESNET_PRETRAINED = RESNET_YOLO_PRETRAINED
    FREEZE_BACKBONE_EPOCHS = RESNET_YOLO_FREEZE_BACKBONE_EPOCHS
    UNFREEZE_LAYER4_EPOCH = RESNET_YOLO_UNFREEZE_LAYER4_EPOCH
    UNFREEZE_LAYER3_EPOCH = RESNET_YOLO_UNFREEZE_LAYER3_EPOCH
    NORMALIZE_IMAGES = RESNET_YOLO_NORMALIZE_IMAGES
    IMAGE_MEAN = RESNET_YOLO_IMAGE_MEAN
    IMAGE_STD = RESNET_YOLO_IMAGE_STD
else:
    raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

EXPERIMENT_NAME = f"{MODEL_NAME}_{DETECTION_HEAD}"

CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, EXPERIMENT_NAME)
OUTPUTS_DIR = os.path.join(OUTPUTS_ROOT, EXPERIMENT_NAME)
LOG_DIR = os.path.join(LOG_ROOT, EXPERIMENT_NAME)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"best_{EXPERIMENT_NAME}.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"last_{EXPERIMENT_NAME}.pth")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, f"{EXPERIMENT_NAME}_history.csv")
