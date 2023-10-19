import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
from segment_anything import sam_model_registry, SamPredictor

curr_dir = os.getcwd()
voc_root = os.path.join(curr_dir, "VOCtrainval_11-May-2012", "VOCdevkit", "VOC2012")

if not os.path.isdir(voc_root):
    raise RuntimeError("VOC Directory is wrong")

splits_dir = os.path.join(voc_root, "ImageSets", "Segmentation")
splits_f = os.path.join(splits_dir, "{}.txt".format("train"))

with open(os.path.join(splits_f)) as f:
    file_names = [x.strip() for x in f.readlines()]

img_dir = os.path.join(voc_root, "JPEGImages")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([2 / 255, 2 / 255, 2 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sp_path = "points_for_sam"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

for file in os.listdir(sp_path):
    file_name = file[:-12]
    print("{}_mask started".format(file_name))

    image = cv2.imread("coco_minitrain2017/{}.jpg".format(file_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    script_dir = os.path.dirname(__file__)
    pt_file = script_dir + "/" + sp_path + "/" + file_name + "_points.json"

    with open(pt_file, "r") as input_file:
        whole_input = json.load(input_file)

    if whole_input == [] or len(whole_input) == 0:
        continue

    semantic_mask = []

    for pt in list(reversed(whole_input)):
        if "seg_point" not in pt:
            print("seg_point not exists")
            continue

        p_pt = np.array([pt["seg_point"]])
        label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=p_pt,
            point_labels=label,
            multimask_output=True,
        )

        mask_list = list(zip(masks, scores))

        best_mask = None

        curr_id = [i for i in coco_80_cats if coco_80_cats[i][0] == pt["category_id"]]
        cat80_id = int(curr_id[0])
        if max(scores) < 0.95:
            best_mask = masks[np.where(scores == max(scores))[0][0]] * cat80_id
        else:
            best_area = 0
            for idx in range(len(mask_list)):
                if mask_list[idx][1] >= 0.95:
                    area = mask_list[idx][0].sum()
                    if area > best_area:
                        best_mask = masks[idx] * cat80_id
                        best_area = area

        semantic_mask.append(best_mask)

    if semantic_mask == []:
        print("no masks")
        continue

    flattened_mask = semantic_mask[0]

    for sm in range(len(semantic_mask) - 1):
        next_layer = semantic_mask[sm + 1]
        flattened_mask[next_layer != 0] = next_layer[next_layer != 0]

    # print(len(semantic_mask))

    # plt.figure(figsize=(10, 10))
    # show_mask(flattened_mask, plt.gca(), random_color=False)
    # plt.axis("off")
    # plt.show()
    np.save("pseudo_labels/{}_mask".format(file_name), flattened_mask)
