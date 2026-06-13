# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/13 14:51
import os
import urllib.request
import tarfile

import config

VOC_URLS = {
    "VOC2007_trainval":
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",

    "VOC2007_test":
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",

    "VOC2012_trainval":
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
}


def download(url, save_path):
    if os.path.exists(save_path):
        print(f"Already exists: {save_path}")
        return

    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved to: {save_path}")


def extract(tar_path, extract_dir):
    print(f"Extracting: {tar_path}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)
    print("Done.")


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    for name, url in VOC_URLS.items():
        filename = url.split("/")[-1]
        tar_path = os.path.join(config.DATA_DIR, filename)

        download(url, tar_path)
        extract(tar_path, config.DATA_DIR)


if __name__ == "__main__":
    main()