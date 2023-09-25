import json
import random
import cv2
import time
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

img_ids = coco.getImgIds()
# print(img_ids)
# print(len(set(img_ids)))

# file_no = 0

for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    ann_data = coco.loadAnns(ann_ids)
    ann_image = coco.loadImgs(img_id)
    print("Creating {} has begun.".format(ann_image[0]["file_name"][:-4]))
    image_path = "train2017/{}".format(ann_image[0]["file_name"])
    # print(image_path)

    # Output path for the new json file
    output_path = "train_pts/{}_points.json".format(ann_image[0]["file_name"][:-4])

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    points = []
    all_polys = []
    for ann in ann_data:
        if 'segmentation' in ann and type(ann['segmentation']) == list:
            for segmentation in ann['segmentation']:

                coords = np.asarray(segmentation).reshape(len(segmentation)//2, 2)

                poly = Polygon(coords.tolist())

                all_polys.append({
                    'image_id': ann['image_id'],
                    'category_id': ann['category_id'],
                    'poly': poly,
                    'poly_area': poly.area,
                    'label': coco.loadCats(ann['category_id'])[0]['name']
                })
    sorted_polys = sorted(all_polys, key=lambda x: x['poly_area'])
    overlap_polys = []
    for p in sorted_polys:
        curr_poly = p['poly']

        if overlap_polys != []:
            for q in overlap_polys:
                try:
                    curr_poly = curr_poly.difference(q)
                except Exception as e:
                    pass

        overlap_polys.append(curr_poly)
        seg_point = curr_poly.representative_point()
        # print(seg_point)
        # print(curr_poly)
        p.pop('poly_area')
        p.pop('poly')
        if not seg_point:
            continue
        point_coord = np.rint(mapping(seg_point)['coordinates'])
        point_coord = point_coord.astype(int).tolist()
        p.update({'seg_point': point_coord})

    try:
        with open(output_path, "w") as output_file:
            json.dump(sorted_polys, output_file)
        print("File {} has been created.".format(ann_image[0]["file_name"][:-4]))
    except Exception as e:
        print(
            f"Error occurred while writing to {ann_image[0]['file_name'][:-4]}: {str(e)}"
        )

    # print(layers)
    # print(points)
