import json
import os
import random
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

# from pycocotools.coco import COCO
# from pycocotools.mask import decode
# from shapely.geometry import Point, Polygon, mapping

curr_dir = os.getcwd()

voc_root = os.path.join(curr_dir, "VOCtrainval_11-May-2012", "VOCdevkit", "VOC2012")

if not os.path.isdir(voc_root):
    raise RuntimeError("VOC Directory is wrong")

splits_dir = os.path.join(voc_root, "ImageSets", "Segmentation")


splits_f = os.path.join(splits_dir, "{}.txt".format("train"))


with open(os.path.join(splits_f)) as f:
    file_names = [x.strip() for x in f.readlines()]

img_dir = os.path.join(voc_root, "JPEGImages")
seg_dir = os.path.join(voc_root, "SegmentationClass")

images = [os.path.join(img_dir, x + ".jpg") for x in file_names]
masks = [os.path.join(seg_dir, x + ".png") for x in file_names]

for image in images:
    img = cv2.imread(image)
    cv2.imshow('image', img)
    break

# # Initializing COCO object
# coco = COCO(ann_path)

# img_ids = coco.getImgIds()


# for img_id in img_ids:
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     ann_data = coco.loadAnns(ann_ids)
#     ann_image = coco.loadImgs(img_id)
#     print("Creating {} has begun.".format(ann_image[0]["file_name"][:-4]))
#     image_path = "coco_minitrain2017{}".format(ann_image[0]["file_name"])

#     # Output path for the new json file
#     output_path = "points_for_sam/{}_points.json".format(ann_image[0]["file_name"][:-4])
    
#     points = []
#     all_polys = []
#     for ann in ann_data:
#         if 'segmentation' in ann and type(ann['segmentation']) == list:
#             for segmentation in ann['segmentation']:

#                 coords = np.asarray(segmentation).reshape(len(segmentation)//2, 2)

#                 poly = Polygon(coords.tolist())

#                 all_polys.append({
#                     'image_id': ann['image_id'],
#                     'category_id': ann['category_id'],
#                     'poly': poly,
#                     'poly_area': poly.area,
#                     'label': coco.loadCats(ann['category_id'])[0]['name']
#                 })
#     sorted_polys = sorted(all_polys, key=lambda x: x['poly_area'])
#     overlap_polys = []
#     for p in sorted_polys:
#         curr_poly = p['poly']

#         if overlap_polys != []:
#             for q in overlap_polys:
#                 try:
#                     curr_poly = curr_poly.difference(q)
#                 except Exception as e:
#                     pass

#         overlap_polys.append(curr_poly)
#         seg_point = curr_poly.representative_point()
#         p.pop('poly_area')
#         p.pop('poly')
#         if not seg_point:
#             continue
#         point_coord = np.rint(mapping(seg_point)['coordinates'])
#         point_coord = point_coord.astype(int).tolist()
#         p.update({'seg_point': point_coord})

#     try:
#         with open(output_path, "w") as output_file:
#             json.dump(sorted_polys, output_file)
#         print("File {} has been created.".format(ann_image[0]["file_name"][:-4]))
#     except Exception as e:
#         print(
#             f"Error occurred while writing to {ann_image[0]['file_name'][:-4]}: {str(e)}"
#         )

