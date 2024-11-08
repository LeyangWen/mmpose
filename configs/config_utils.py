from _base_.datasets.VEHS7M_37kpts import dataset_info as dataset_info_VEHS7M_37kpts
from _base_.datasets.coco_wholebody import dataset_info as dataset_info_coco_wholebody
from _base_.datasets.mpii import dataset_info as dataset_info_mpii
from _base_.datasets.mpii_trb import dataset_info as dataset_info_mpii_trb
from _base_.datasets.aic import dataset_info as dataset_info_aic

import numpy as np

def mapping_convert(map_ab, map_bc):
    """
    note: if a have keypoints not in b but in c, it will not be converted
    Args:
        map_ab:
        map_bc:
    Returns:
        map_ac:
    """
    map_ac = []
    map_ab_check = []
    for ab_pair in map_ab:
        a_id = ab_pair[0]
        b_id = ab_pair[1]
        c_id = False
        map_ab_check.append(a_id)
        for bc_pair in map_bc:
            if bc_pair[0] == b_id:
                c_id = bc_pair[1]
        if c_id:
            map_ac.append((a_id, c_id))
    map_ab_check = set(map_ab_check)
    missing_id = []
    for check_id in range(max(map_ab_check)):
        if check_id not in map_ab_check:
            missing_id.append(check_id)
    if len(missing_id)>0:
        print(f"Mapping_convert: missing id in dataset_a, check if they are in c: {missing_id}")
    print(map_ac)
    return map_ac

def mapping_check(map, dataset_info_a, dataset_info_b):
    """
    Args:
        map: list of tuple
        dataset_info_a: dict (e.g., configs/_base_/datasets/VEHS7M_37kpts.py)
        dataset_info_b: dict
    Returns:
        None
    """
    print("Mapping_check: ")
    for map_pair in map:
        a_id = map_pair[0]
        b_id = map_pair[1]
        print(f"{a_id}:{dataset_info_a['keypoint_info'][a_id]['name']} -> {b_id}:{dataset_info_b['keypoint_info'][b_id]['name']}")
    return None



# Step 1: copy mapping from configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7Mplus-384x288.py
# mapping
# (AIC-kpts, COCO133-kpts)
aic_coco133 = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12),
               (7, 14), (8, 16), (9, 11), (10, 13), (11, 15)]

mpii_coco133 = [
    (0, 16),
    (1, 14),
    (2, 12),
    (3, 11),
    (4, 13),
    (5, 15),
    (10, 10),
    (11, 8),
    (12, 6),
    (13, 5),
    (14, 7),
    (15, 9),
]

# convert
coco133_VEHS7M = [(11-1, 1),
                    (10-1, 2),
                    (13-1, 3),
                    (12-1, 4),
                    (15-1, 5),
                    (14-1, 6),
                    (17-1, 7),
                    (16-1, 8),
                    (21-1, 9),
                    (18-1, 10),
                    (122-1, 11),
                    (101-1, 12),
                    (9-1, 13),
                    (8-1, 14),
                    (7-1, 15),
                    (6-1, 16),
                    (51-1, 17),
                    (5-1, 20),
                    (4-1, 21),
                    (118-1, 33),
                    (130-1, 34),
                    (97-1, 35),
                    (109-1, 36)]
# ref: configs/_base_/datasets/VEHS7M_37kpts.py


print(f"COCO-WB vs. VEHS7M-37kpts")
mapping_check(coco133_VEHS7M, dataset_info_coco_wholebody, dataset_info_VEHS7M_37kpts)
print("#"*40)


print(f"AIC vs. VEHS7M-37kpts")
aic_VEHS7M = mapping_convert(aic_coco133, coco133_VEHS7M)
mapping_check(aic_VEHS7M, dataset_info_aic, dataset_info_VEHS7M_37kpts)
#aic_VEHS7M = [(0, 15), (1, 13), (2, 1), (3, 16), (4, 14), (5, 2), (6, 3), (7, 5), (8, 7), (9, 4), (10, 6), (11, 8)]
print("#"*40)

print(f"MPII-TRB vs. VEHS7M-37kpts")
mpii_trb_VEHS7M = [(0, 16), (1, 15),
                   (2, 14), (3, 13),
                   (4, 2), (5, 1),
                   (6, 4), (7, 3),
                   (8, 6), (9, 5),
                   (10, 8), (11, 7),
                   (12, 19),
                   (13, 22),
                   (18, 30), (19, 29),
                   (20, 32),(21, 31)]

mapping_check(mpii_trb_VEHS7M, dataset_info_mpii_trb, dataset_info_VEHS7M_37kpts)
print("#"*40)


# do the checks



