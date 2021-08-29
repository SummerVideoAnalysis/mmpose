import cv2
import h5py
import json
import numpy as np
import os
import time
from os.path import join


# get img file name from ascii code
def get_name(name_ascii):
    text = ""
    for i in name_ascii:
        text += f"{chr(i)}"
        if ".jpg" in text:
            break

    return text


def save_coco_anno(
    keypoints_all,
    imgs_all,
    keypoints_info,
    skeleton_info,
    dataset,
    img_root,
    save_path,
    start_img_id=0,
    start_ann_id=0,
):
    """Save annotations in coco-format.
    :param keypoints_all: keypoint annotations.
    :param annotated_all: images annotated or not.
    :param imgs_all: the array of images.
    :param keypoints_info: infomation about keypoint name.
    :param skeleton_info: infomation about skeleton connection.
    :param dataset: infomation about dataset name.
    :param img_root: the path to save images.
    :param save_path: the path to save transformed annotation file.
    :param start_img_id: the starting point to count the image id.
    :param start_ann_id: the starting point to count the annotation id.
    """
    images = []
    annotations = []

    print(len(imgs_all))
    img_id = start_img_id
    ann_id = start_ann_id

    num_annotations, keypoints_num, _ = keypoints_all.shape

    for i in range(num_annotations):
        img_name = imgs_all[i]
        img = cv2.imread(f"{join(img_root, img_name)}")
        keypoints = keypoints_all[i]

        min_x = np.min(keypoints[:, 0])
        min_y = np.min(keypoints[:, 1])
        max_x = np.max(keypoints[:, 0])
        max_y = np.max(keypoints[:, 1])

        anno = {}

        kps = []
        for i, j in keypoints:
            kps.extend([i, j, 1])

        anno["keypoints"] = kps
        anno["image_id"] = img_id
        anno["id"] = ann_id
        anno["num_keypoints"] = 18
        anno["bbox"] = [
            float(min_x),  # originX
            float(min_y),  # originY
            float(max_x - min_x + 1),  # width
            float(max_y - min_y + 1),  # height
        ]
        anno["iscrowd"] = 0
        anno["area"] = anno["bbox"][2] * anno["bbox"][3]  # width * height
        anno["category_id"] = 1  # 1 for human

        annotations.append(anno)
        ann_id += 1

        image = {}
        image["id"] = img_id
        image["file_name"] = img_name
        image["height"] = img.shape[0]
        image["width"] = img.shape[1]

        images.append(image)
        img_id += 1

        # cv2.imwrite(os.path.join(img_root, image['file_name']), img)

    cocotype = {}

    cocotype["info"] = {}
    cocotype["info"]["description"] = "VIP-HARPET dataset"
    cocotype["info"]["version"] = "1.0"
    cocotype["info"]["year"] = time.strftime("%Y", time.localtime())
    cocotype["info"]["date_created"] = time.strftime("%Y/%m/%d", time.localtime())

    cocotype["images"] = images
    cocotype["annotations"] = annotations
    cocotype["categories"] = [
        {
            "supercategory": "human",
            "id": 1,
            "name": "augmented_human",
            "keypoints": keypoints_info,
            "skeleton": skeleton_info,
        }
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(cocotype, open(save_path, "w"), indent=4)
    print("number of images:", img_id)
    print("number of annotations:", ann_id)
    print(f"done {save_path}")


keypoints_info = [
    "rightAnkle",
    "rightKnee",
    "rightHip",
    "leftHip",
    "leftKnee",
    "leftAnkle",
    "pelvis",
    "thorax",
    "neck",
    "head",
    "rightWrist",
    "rightElbow",
    "rightShoulder",
    "leftShoulder",
    "leftElbow",
    "leftWrist",
    "hockeyGrip",
    "hockeyHeel",
]
dataset_dir = "."

with h5py.File(os.path.join("annot_val.h5"), "r") as f:
    # List all groups
    annotations = np.array(f["part"])
    images = []
    for i in range(annotations.shape[0]):
        images.append(get_name(list(map(int, f["imgname"][i]))))

# src: https://github.com/jgraving/DeepPoseKit-Data/blob/master/datasets/human/mpii/mpii_example_loading.ipynb
# skeleton_info = np.array([1, 2, 6, 6, 3, 4, 7, 8, 9, -1, 11, 12, 8, 8, 13, 14, 10, 16])
# NOTE that it is 1-indexed for some reason. Saw 1-indexed structure in the test annotations in the repository
skeleton_info = [
    [10, 9],
    [13, 9],
    [14, 9],
    [13, 12],
    [12, 11],
    [11, 17],  # right-handed
    # [16, 17],   # left-handed
    [17, 18],
    [13, 8],
    [14, 8],
    [8, 7],
    [7, 3],
    [7, 4],
    [3, 2],
    [2, 1],
    [4, 5],
    [5, 6],
    [14, 15],
    [15, 16],
]
# annotation_num, kpt_num, _ = annotations.shape
dataset = "harpet"

img_root = "val"
save_coco_anno(
    annotations,
    images,
    keypoints_info,
    skeleton_info,
    dataset,
    img_root,
    os.path.join("annotations", f"{dataset}_val.json"),
)
