import json
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.mask import decode
from shapely.geometry import Point, Polygon, mapping

# Path to the COCO annotations file
ann_path = "annotations_trainval2017/annotations/instances_train2017.json"
img_path = "train2017"

# Initializing COCO object
coco = COCO(ann_path)

img_ids = coco.getImgIds()[0]
print(len(img_ids))
# print(len(set(img_ids)))

file_no = 0

for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    ann_data = coco.loadAnns(ann_ids)
    ann_image = coco.loadImgs(img_ids)
    image_path = "train2017/{}".format(ann_image[0]["file_name"])
    # print(image_path)

    # Output path for the new json file
    output_path = "train_pts/{}_points.txt".format(ann_image[0]["file_name"][:-4])

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    points = []
    for ann in ann_data:
        if "segmentation" in ann and type(ann["segmentation"]) == list:
            for segmentation in ann["segmentation"]:
                # single_mask = coco.annToMask(ann)
                # plt.imshow(single_mask)
                # label_name = coco.loadCats(ann['category_id'])[0]['name']
                # print(label_name)
                # layers += 1
                coords = np.asarray(segmentation).reshape(len(segmentation) // 2, 2)
                # # print(coords.tolist())
                poly = Polygon(coords.tolist())
                center_point = poly.centroid
                # print(center_point)
                # min_x, min_y, max_x, max_y = poly.bounds

                # while True:
                #     random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
                #     if (random_point.within(poly)):
                point_coord = mapping(center_point)["coordinates"]
                points.append(
                    {
                        "image_id": ann["image_id"],
                        "category_id": ann["category_id"],
                        "point": point_coord,
                        "label": coco.loadCats(ann["category_id"])[0]["name"],
                    }
                )

                #         break
        try:
            with open(output_path, "w") as output_file:
                json.dump(points, output_file)
            print("File {} has been created.".format(ann_image[0]["file_name"][:-4]))
        except Exception as e:
            print(
                f"Error occurred while writing to {ann_image[0]['file_name'][:-4]}: {str(e)}"
            )


    # print(layers)
    # print(points)
