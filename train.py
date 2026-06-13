# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/13 12:08
import torch
from torch import optim
from torch.utils.data import ConcatDataset, random_split, DataLoader

import config
from model.yolov1 import YOLOv1
from utils import loss
from utils.dataset import VOCDataset


def get_datasets():
    dataset_07 = VOCDataset(root_dir=config.DATA_DIR, year="2007", image_set='trainval',
                            S=7, B=2, C=20, img_size=448)
    dataset_12 = VOCDataset(root_dir=config.DATA_DIR, year="2012", image_set='trainval',
                            S=7, B=2, C=20, img_size=448)

    full_dataset =  ConcatDataset([dataset_07, dataset_12])

    # 分割
    train_size = int(config.DATASET_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            total_loss += loss.item()

    return total_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS)

    model = YOLOv1(in_channels=3, split_size=7, num_boxes=2, num_classes=20).to(device)

    optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=0.9, weight_decay=5e-4)

    criterion = loss.YoloLoss()

    best_val_loss = float("inf")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:,4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, config.LAST_MODEL_PATH)

if __name__ == "__main__":
    main()











