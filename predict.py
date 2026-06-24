import os
import random

import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

import config
from model.factory import build_yolo_model
from utils.dataset import VOCDataset, VOC_CLASSES
from utils.checkpoint import load_model_state
from utils.decode import decode_prediction
from utils.visualize import draw_boxes


def load_model(device):
    model = build_yolo_model(config).to(device)

    ckpt_path = config.BEST_MODEL_PATH
    if not os.path.exists(ckpt_path):
        ckpt_path = config.LAST_MODEL_PATH

    # map_location=device 确保不管这个模型之前是在谁的服务器、哪张显卡上训出来的，
    # 都能安全地被拉回你当前的 device（比如你的笔记本 CPU），防止显卡不匹配报错。
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
    dataset = VOCDataset(root_dir=config.DATA_DIR, year="2007", image_set="test",
                         S=config.S, B=config.B, C=config.NUM_CLASSES, img_size=config.IMG_SIZE, train=False)

    #  随机生成几个数，对应处理的图片编号
    indices = random.sample(range(len(dataset)), k=5)
    for idx, data_idx in enumerate(indices):
        image_id = dataset.ids[data_idx]
        image_path = os.path.join(dataset.image_dir, image_id + ".jpg")
        image = Image.open(image_path).convert("RGB")
        input_img = TF.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        # TF进行维度大翻转和归一化，unsqueeze(0)：强行升维，伪造“批次（Batch）”
        input_tensor = TF.to_tensor(input_img).unsqueeze(0).to(device)
        if config.NORMALIZE_IMAGES:
            input_tensor = TF.normalize(
                input_tensor.squeeze(0),
                mean=config.IMAGE_MEAN,
                std=config.IMAGE_STD,
            ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_tensor)[0]

        # 对模型输出进行解码，即恢复位置、类预测、置信度
        boxes = decode_prediction(pred, conf_threshold=0.3)
        save_path = os.path.join(config.OUTPUTS_DIR, f"pred_{idx}_{image_id}.jpg")
        # 画框
        draw_boxes(image.copy(), boxes, save_path)
        print(f"{image_id}: {len(boxes)} boxes -> {save_path}")


if __name__ == "__main__":
    main()
