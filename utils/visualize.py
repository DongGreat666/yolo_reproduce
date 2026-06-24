# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/13 12:40

from PIL import ImageDraw

from utils.dataset import VOC_CLASSES


def draw_boxes(image, boxes, save_path):
    draw = ImageDraw.Draw(image)
    w_img, h_img = image.size

    for class_id, score, x1, y1, x2, y2 in boxes:
        x1 = int(x1 * w_img)
        y1 = int(y1 * h_img)
        x2 = int(x2 * w_img)
        y2 = int(y2 * h_img)
        label = f"{VOC_CLASSES[class_id]} {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], width=3, outline='black')
        draw.text((x1, max(0, y1 - 12)), label, fill='black')

    image.save(save_path)


