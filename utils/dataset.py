import os
import random
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import config


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class VOCDataset(Dataset):
    def __init__(
        self,
        root_dir,
        year="2007",
        image_set="trainval",
        train=True,
        S=7,
        B=2,
        C=20,
        img_size=448,
        transform=None,
    ):
        self.root_dir = root_dir
        self.year = year
        self.image_set = image_set
        self.train = train
        self.S = S
        self.B = B
        self.C = C
        self.img_size = img_size
        self.transform = transform

        self.voc_dir = os.path.join(root_dir, "VOCdevkit", f"VOC{year}")
        self.image_dir = os.path.join(self.voc_dir, "JPEGImages")
        self.anno_dir = os.path.join(self.voc_dir, "Annotations")
        split_file = os.path.join(self.voc_dir, "ImageSets", "Main", f"{image_set}.txt")

        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}
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
        boxes = self.parse_voc_xml(anno_path)

        image = TF.resize(image, (self.img_size, self.img_size))

        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                for box in boxes:
                    box[1] = 1.0 - box[1]

            image, boxes = self.random_scale_translate(image, boxes)
            image = self.color_jitter(image)

        image = TF.to_tensor(image)
        if config.NORMALIZE_IMAGES:
            image = TF.normalize(image, mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        label = self.encode_label(boxes)
        return image, label

    def parse_voc_xml(self, anno_path):
        tree = ET.parse(anno_path)
        root = tree.getroot()

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        boxes = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_idx = self.class_to_idx[class_name]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            boxes.append([class_idx, x_center, y_center, w, h])

        return boxes

    def random_scale_translate(self, image, boxes):
        scale = random.uniform(0.8, 1.2)
        new_size = max(1, int(round(self.img_size * scale)))
        resized = TF.resize(image, (new_size, new_size))

        max_shift = int(round(0.2 * self.img_size))
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)

        canvas = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        canvas.paste(resized, (dx, dy))

        adjusted = []
        for class_idx, x, y, w, h in boxes:
            x1 = (x - w / 2) * self.img_size * scale + dx
            y1 = (y - h / 2) * self.img_size * scale + dy
            x2 = (x + w / 2) * self.img_size * scale + dx
            y2 = (y + h / 2) * self.img_size * scale + dy

            x1 = max(0.0, min(float(self.img_size), x1))
            y1 = max(0.0, min(float(self.img_size), y1))
            x2 = max(0.0, min(float(self.img_size), x2))
            y2 = max(0.0, min(float(self.img_size), y2))

            bw = x2 - x1
            bh = y2 - y1
            if bw < 1 or bh < 1:
                continue

            adjusted.append([
                class_idx,
                ((x1 + x2) / 2) / self.img_size,
                ((y1 + y2) / 2) / self.img_size,
                bw / self.img_size,
                bh / self.img_size,
            ])

        return canvas, adjusted

    def encode_label(self, boxes):
        label = torch.zeros((self.S, self.S, self.C + 5))

        for class_idx, x, y, w, h in boxes:
            i = min(int(self.S * y), self.S - 1)
            j = min(int(self.S * x), self.S - 1)

            x_cell = self.S * x - j
            y_cell = self.S * y - i

            if label[i, j, self.C] == 0:
                label[i, j, self.C] = 1
                label[i, j, self.C + 1:self.C + 5] = torch.tensor([x_cell, y_cell, w, h])
                label[i, j, int(class_idx)] = 1

        return label


if __name__ == "__main__":
    dataset = VOCDataset(
        root_dir=config.DATA_DIR,
        year="2007",
        image_set="trainval",
        S=7,
        B=2,
        C=20,
        img_size=448,
    )
    image, label = dataset[0]
    print(image.shape)
    print(label.shape)
