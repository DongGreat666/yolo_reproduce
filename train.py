import csv
import os

import torch
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import config
from model.factory import build_yolo_model
from model.resnet_yolo import ResNetYOLO
from utils.loss import YoloLoss
from utils.dataset import VOCDataset
from utils.metrics import evaluate_map
from utils.checkpoint import load_model_state, model_state_dict, unwrap_model


csv_file = config.TRAIN_LOG_PATH


def get_datasets():
    train_07 = VOCDataset(
        root_dir=config.DATA_DIR,
        year="2007",
        image_set="trainval",
        train=True,
        S=config.S,
        B=config.B,
        C=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
    )
    train_12 = VOCDataset(
        root_dir=config.DATA_DIR,
        year="2012",
        image_set="trainval",
        train=True,
        S=config.S,
        B=config.B,
        C=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
    )
    val_07 = VOCDataset(
        root_dir=config.DATA_DIR,
        year="2007",
        image_set="test",
        train=False,
        S=config.S,
        B=config.B,
        C=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
    )
    return ConcatDataset([train_07, train_12]), val_07


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(tqdm(loader)):
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss = criterion(preds, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss at train batch {batch_idx}")
            return float("nan")

        optimizer.zero_grad()
        loss.backward()
        if config.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    max_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf val loss at batch {batch_idx}")
                return float("nan")

            max_loss = max(max_loss, loss.item())
            total_loss += loss.item()

    print("max val batch loss:", max_loss)
    return total_loss / len(loader)


def adjust_lr(optimizer, epoch):
    if epoch < config.WARMUP_EPOCHS:
        progress = (epoch + 1) / config.WARMUP_EPOCHS
        lr = config.LR_WARMUP_START + progress * (
            config.LR_WARMUP_END - config.LR_WARMUP_START
        )
    elif epoch < config.WARMUP_EPOCHS + config.LR_STAGE1_EPOCHS:
        lr = config.LR_WARMUP_END
    elif epoch < config.WARMUP_EPOCHS + config.LR_STAGE1_EPOCHS + config.LR_STAGE2_EPOCHS:
        lr = config.LR_WARMUP_END
    else:
        lr = config.LR_WARMUP_END

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def apply_training_stage(model, epoch):
    core_model = unwrap_model(model)
    if not isinstance(core_model, ResNetYOLO):
        return "yolov1-all"

    if epoch < config.FREEZE_BACKBONE_EPOCHS:
        core_model.freeze_backbone()
        return "resnet_yolo-freeze-backbone"
    if epoch < config.UNFREEZE_LAYER3_EPOCH:
        core_model.unfreeze_layer4()
        return "resnet_yolo-unfreeze-layer4"

    core_model.unfreeze_layer3_layer4()
    return "resnet_yolo-unfreeze-layer3-layer4"


def write_log(epoch, train_loss, val_loss, map50, start_epoch):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    append_log = config.RESUME and start_epoch > 0
    file_mode = "a" if append_log or epoch > start_epoch else "w"
    file_exists = os.path.exists(csv_file) and file_mode == "a"

    with open(csv_file, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "map50"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "map50": "" if map50 is None else map50,
            }
        )


def load_pretrained_backbone(model, device):
    backbone_path = getattr(config, "BACKBONE_INIT_PATH", "") or config.PRETRAINED_BACKBONE_PATH
    if not backbone_path:
        return
    if not os.path.exists(backbone_path):
        raise FileNotFoundError(backbone_path)

    checkpoint = torch.load(backbone_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "backbone_state_dict" in checkpoint:
        state_dict = checkpoint["backbone_state_dict"]
    else:
        state_dict = checkpoint

    backbone_state = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]

        if clean_key.startswith("darknet."):
            backbone_state[clean_key[len("darknet."):]] = value
        elif clean_key.startswith("backbone."):
            backbone_state[clean_key[len("backbone."):]] = value
        elif clean_key.startswith("layers."):
            backbone_state[clean_key] = value

    if not backbone_state:
        raise RuntimeError(f"No backbone weights found in {backbone_path}")

    core_model = unwrap_model(model)
    if hasattr(core_model, "darknet"):
        target_backbone = core_model.darknet
    elif hasattr(core_model, "backbone"):
        target_backbone = core_model.backbone
    else:
        raise RuntimeError("Model has no darknet/backbone module")

    missing, unexpected = target_backbone.load_state_dict(backbone_state, strict=False)
    print(
        "Loaded pretrained backbone:",
        backbone_path,
        f"(missing={len(missing)}, unexpected={len(unexpected)})",
    )
    if missing:
        print("  missing sample:", missing[:5])
    if unexpected:
        print("  unexpected sample:", unexpected[:5])


def maybe_wrap_data_parallel(model):
    if not config.USE_DATA_PARALLEL:
        return model
    if not torch.cuda.is_available():
        print("DataParallel requested but CUDA is unavailable; using single device.")
        return model
    if torch.cuda.device_count() < 2:
        print(f"DataParallel requested but only {torch.cuda.device_count()} GPU(s) found.")
        return model

    valid_gpu_ids = [gpu_id for gpu_id in config.GPU_IDS if gpu_id < torch.cuda.device_count()]
    if len(valid_gpu_ids) < 2:
        print(f"Need at least 2 valid GPU ids for DataParallel, got {valid_gpu_ids}.")
        return model

    print(f"Using DataParallel on GPU ids: {valid_gpu_ids}")
    return torch.nn.DataParallel(model, device_ids=valid_gpu_ids)


def main():
    device = torch.device(config.DEVICE)
    print("Using Device:", device)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    train_dataset, val_dataset = get_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_yolo_model(config).to(device)
    load_pretrained_backbone(model, device)
    model = maybe_wrap_data_parallel(model)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=0.9,
        weight_decay=5e-4,
    )
    criterion = YoloLoss(
        S=config.S,
        B=config.B,
        C=config.NUM_CLASSES,
        lambda_coord=config.LAMBDA_COORD,
        lambda_noobj=config.LAMBDA_NOOBJ,
        debug=config.LOSS_DEBUG,
        debug_interval=config.LOSS_DEBUG_INTERVAL,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if config.RESUME and os.path.exists(config.LAST_MODEL_PATH):
        checkpoint = torch.load(config.LAST_MODEL_PATH, map_location=device)
        load_model_state(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"Resume from epoch {start_epoch}")
    else:
        print("Start training from scratch")

    for epoch in range(start_epoch, config.EPOCHS):
        stage = apply_training_stage(model, epoch)
        current_lr = adjust_lr(optimizer, epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        map50 = None

        if (
            torch.isnan(torch.tensor(train_loss))
            or torch.isnan(torch.tensor(val_loss))
            or torch.isinf(torch.tensor(train_loss))
            or torch.isinf(torch.tensor(val_loss))
        ):
            print("NaN/Inf detected, skip saving checkpoint and stop training.")
            break

        should_eval_map = (
            config.MAP_EVAL_INTERVAL > 0
            and ((epoch + 1) % config.MAP_EVAL_INTERVAL == 0 or epoch + 1 == config.EPOCHS)
        )
        if should_eval_map:
            map50, ap_by_class = evaluate_map(
                model,
                val_dataset,
                device,
                iou_threshold=config.MAP_IOU_THRESHOLD,
                conf_threshold=config.MAP_CONF_THRESHOLD,
                nms_threshold=config.MAP_NMS_THRESHOLD,
            )
            print(f"mAP@{config.MAP_IOU_THRESHOLD:.2f}: {map50:.4f}")

        map_text = "-" if map50 is None else f"{map50:.4f}"
        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"LR: {current_lr:.6f} "
            f"Stage: {stage} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"mAP50: {map_text}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_state_dict(model), config.BEST_MODEL_PATH)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            config.LAST_MODEL_PATH,
        )
        write_log(epoch, train_loss, val_loss, map50, start_epoch)

    print("Training finished")


if __name__ == "__main__":
    main()
