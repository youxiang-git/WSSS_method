import opencv
import json
from pycocotools.coco import COCO
from pycocotools.mask import decode, frPyObjects
import matplotlib.pyplot as plt
import numpy as np

ann_path = "annotations/instances_minitrain2017.json"
img_path = "coco_minitrain2017"

with open("all_cats.json", "r") as file:
    coco_80_cats = json.load(file)

coco = COCO(ann_path)

img_ids = coco.getImgIds()

for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    ann_data = coco.loadAnns(ann_ids)
    ann_image = coco.loadImgs(img_id)
    if ann_data == []:
        print("{} has no masks".format(ann_image[0]["file_name"]))
        continue
    image_path = "{}/{}".format(img_path, ann_image[0]["file_name"])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, c = image.shape

    ann_data_sorted = sorted(ann_data, key=lambda x: x["area"], reverse=True)

    masks = []
    for ann in ann_data_sorted:
        curr_id = [i for i in coco_80_cats if coco_80_cats[i][0] == ann["category_id"]]
        cat80_id = int(curr_id[0])
        if "segmentation" in ann and type(ann["segmentation"]) == list:
            for segmentation in ann["segmentation"]:
                rle = frPyObjects([segmentation], height, width)
                mask = decode(rle)
                mask = mask * cat80_id
                masks.append(mask)

    flattened_mask = masks[0]

    for sm in range(len(masks) - 1):
        next_layer = masks[sm + 1]
        flattened_mask[next_layer != 0] = next_layer[next_layer != 0]

    np.save(
        "coco_minitrain2017_masks/{}_masks".format(ann_image[0]["file_name"][:-4]),
        flattened_mask.squeeze(2),
    )
