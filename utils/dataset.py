# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/13 12:39
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import xml.etree.ElementTree as ET

import config

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class VOCDataset(Dataset):
    def __init__(self, root_dir, year="2007", image_set="trainval",
                 S=7, B=2, C=20, img_size=448, transform=None):
        self.root_dir = root_dir
        self.year = year
        self.image_set = image_set
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

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]

        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        anno_path = os.path.join(self.anno_dir, image_id + ".xml")

        image = Image.open(image_path).convert("RGB")
        boxes = self.parse_voc_xml(anno_path)

        image = self.transform(image)
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

            # 转成相应坐标
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            boxes.append([class_idx, x_center, y_center, w, h])

        return boxes

    def encode_label(self, boxes):
        """
        输出 label shape: [S, S, C + 5]
        注意：YOLOv1 label 里通常只存一个 bbox。预测是 B=2 个 bbox，但 target 只需要一个真实框。
        """
        label = torch.zeros((self.S, self.S, self.C + 5))

        for box in boxes:
            class_idx, x, y, w, h = box

            i = int(self.S * y) # row
            j = int(self.S * x) # col

            i = min(i, self.S - 1)
            j = min(j, self.S - 1)

            x_cell = self.S * x - j
            y_cell = self.S * y - i

            if label[i, j, self.C] == 0:
                label[i, j, self.C] = 1
                label[i, j, self.C + 1:self.C + 5] = torch.tensor([
                    x_cell, y_cell, w, h
                ])
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
        img_size=448
    )

    image, label = dataset[0]
    print(image.shape)
    print(label.shape)









