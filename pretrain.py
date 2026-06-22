import argparse
import os
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

import config
from model.yolov1 import DarknetClassifier
from utils.checkpoint import unwrap_model
from utils.dataset import VOC_CLASSES


class VOCClassificationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        year="2007",
        image_set="trainval",
        train=True,
        img_size=448,
        normalize=False,
        image_mean=None,
        image_std=None,
    ):
        self.root_dir = root_dir
        self.year = year
        self.image_set = image_set
        self.train = train
        self.img_size = img_size
        self.normalize = normalize
        self.image_mean = image_mean or [0.0, 0.0, 0.0]
        self.image_std = image_std or [1.0, 1.0, 1.0]

        self.voc_dir = os.path.join(root_dir, "VOCdevkit", f"VOC{year}")
        self.image_dir = os.path.join(self.voc_dir, "JPEGImages")
        self.anno_dir = os.path.join(self.voc_dir, "Annotations")
        split_file = os.path.join(self.voc_dir, "ImageSets", "Main", f"{image_set}.txt")

        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(VOC_CLASSES)}
        self.color_jitter = transforms.ColorJitter(
            brightness=0.5,
            contrast=0.2,
            saturation=0.5,
            hue=0.05,
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        anno_path = os.path.join(self.anno_dir, image_id + ".xml")

        image = Image.open(image_path).convert("RGB")
        target = self.parse_multihot_label(anno_path)

        image = TF.resize(image, (self.img_size, self.img_size))
        if self.train:
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
            image = self.color_jitter(image)

        image = TF.to_tensor(image)
        if self.normalize:
            image = TF.normalize(image, mean=self.image_mean, std=self.image_std)
        return image, target

    def parse_multihot_label(self, anno_path):
        target = torch.zeros(len(VOC_CLASSES), dtype=torch.float32)
        tree = ET.parse(anno_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_idx = self.class_to_idx[class_name]
            target[class_idx] = 1.0

        return target


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Darknet backbone on VOC multi-label classification.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join(config.CHECKPOINT_ROOT, "yolov1"),
    )
    return parser.parse_args()


def get_datasets(args):
    train_07 = VOCClassificationDataset(
        root_dir=config.DATA_DIR,
        year="2007",
        image_set="trainval",
        train=True,
        img_size=args.img_size,
        normalize=config.YOLOV1_NORMALIZE_IMAGES,
        image_mean=config.YOLOV1_IMAGE_MEAN,
        image_std=config.YOLOV1_IMAGE_STD,
    )
    train_12 = VOCClassificationDataset(
        root_dir=config.DATA_DIR,
        year="2012",
        image_set="trainval",
        train=True,
        img_size=args.img_size,
        normalize=config.YOLOV1_NORMALIZE_IMAGES,
        image_mean=config.YOLOV1_IMAGE_MEAN,
        image_std=config.YOLOV1_IMAGE_STD,
    )
    val_07 = VOCClassificationDataset(
        root_dir=config.DATA_DIR,
        year="2007",
        image_set="test",
        train=False,
        img_size=args.img_size,
        normalize=config.YOLOV1_NORMALIZE_IMAGES,
        image_mean=config.YOLOV1_IMAGE_MEAN,
        image_std=config.YOLOV1_IMAGE_STD,
    )
    return ConcatDataset([train_07, train_12]), val_07


def maybe_wrap_data_parallel(model):
    if not config.USE_DATA_PARALLEL:
        return model
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return model

    valid_gpu_ids = [gpu_id for gpu_id in config.GPU_IDS if gpu_id < torch.cuda.device_count()]
    if len(valid_gpu_ids) < 2:
        return model
    print(f"Using DataParallel on GPU ids: {valid_gpu_ids}")
    return torch.nn.DataParallel(model, device_ids=valid_gpu_ids)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(loader):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        if config.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def average_precision(scores, targets):
    if targets.sum().item() == 0:
        return None

    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order]
    tp = sorted_targets
    fp = 1 - sorted_targets

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = tp_cumsum / (targets.sum() + 1e-6)
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


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    scores = torch.sigmoid(logits)

    ap_values = []
    for class_idx in range(targets.shape[1]):
        ap = average_precision(scores[:, class_idx], targets[:, class_idx])
        if ap is not None:
            ap_values.append(ap)

    mean_ap = sum(ap_values) / max(1, len(ap_values))
    return total_loss / len(loader), mean_ap


def save_checkpoint(model, optimizer, epoch, best_map, path):
    core_model = unwrap_model(model)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": core_model.state_dict(),
            "backbone_state_dict": core_model.darknet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_map": best_map,
        },
        path,
    )


def main():
    args = parse_args()
    device = torch.device(config.DEVICE)
    print("Using Device:", device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    last_path = os.path.join(args.checkpoint_dir, "darknet_cls_last.pth")
    best_path = os.path.join(args.checkpoint_dir, "darknet_cls_best.pth")
    backbone_path = os.path.join(args.checkpoint_dir, "darknet_cls_backbone.pth")

    train_dataset, val_dataset = get_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DarknetClassifier(num_classes=config.NUM_CLASSES).to(device)
    model = maybe_wrap_data_parallel(model)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_map = 0.0
    if args.resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint.get("best_map", best_map)
        print(f"Resume from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_map = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"LR: {args.lr:.6f} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Cls mAP: {val_map:.4f}"
        )

        save_checkpoint(model, optimizer, epoch + 1, best_map, last_path)
        if val_map > best_map:
            best_map = val_map
            save_checkpoint(model, optimizer, epoch + 1, best_map, best_path)
            torch.save(unwrap_model(model).darknet.state_dict(), backbone_path)

    print("Classification pretraining finished")
    print("Best backbone:", backbone_path)


if __name__ == "__main__":
    main()
