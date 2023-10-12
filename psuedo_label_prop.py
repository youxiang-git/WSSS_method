import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
from segment_anything import sam_model_registry, SamPredictor


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
    file_name = "000000581466"

    print(file_name)

    image = cv2.imread("train2017/{}.jpg".format(file_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    script_dir = os.path.dirname(__file__)
    pt_file = script_dir + "/" + sp_path + "/" + file_name + "_points.json"

    with open(pt_file, "r") as input_file:
        whole_input = json.load(input_file)

    semantic_mask = []

    for pt in list(reversed(whole_input)):
        print(pt)
        p_pt = np.array([pt["seg_point"]])
        label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=p_pt,
            point_labels=label,
            multimask_output=True,
        )

        mask_list = list(zip(masks, scores))

        best_mask = None

        if max(scores) < 0.95:
            print("hello im here")
            best_mask = masks[np.where(scores == max(scores))[0][0]] * pt["category_id"]
        else:
            best_area = 0
            for idx in range(len(mask_list)):
                if mask_list[idx][1] >= 0.95:
                    area = mask_list[idx][0].sum()
                    if area > best_area:
                        best_mask = masks[idx] * pt["category_id"]
                        best_area = area

        semantic_mask.append(best_mask)

    flattened_mask = semantic_mask[0]

    for sm in range(len(semantic_mask) - 1):
        next_layer = semantic_mask[sm + 1]
        flattened_mask[next_layer != 0] = next_layer[next_layer != 0]

    print(len(semantic_mask))

    plt.figure(figsize=(10, 10))
    show_mask(flattened_mask, plt.gca(), random_color=False)
    plt.axis("off")
    plt.show()

    break