# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/13 12:08

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据集
DATA_DIR = "./data"
DATASET_RATIO = 0.9
IMG_SIZE = 448
NUM_CLASSES = 20
S = 7
B = 2


# 训练参数
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0

# loss
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5

# path
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
OUTPUT_DIR = "./outputs"

BEST_MODEL_PATH = "./checkpoints/best_yolov1.pth"
LAST_MODEL_PATH = "./checkpoints/last_yolov1.pth"
TRAIN_LOG_PATH = "./logs/train_history.csv"


