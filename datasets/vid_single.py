# Modified by Sairam VCR and Pranoy Panda
# ------------------------------------------------------------------------
# Modified from TransVOD_lite
# Copyright (c) Shanghai Jiao Tong University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

"""

import sys

from pathlib import Path
import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
from torch.utils.data.dataset import ConcatDataset
import random
import math
import numpy as np
import copy

import datasets.transforms as T 
from .coco_video_parser import CocoVID



def ChooseFrame(List, Gap, num_frames):
    ret = []
    start_id = 0
    max_gap = Gap*num_frames
    num = len(List) // max_gap
    for i in range(num):
        start_id = i * max_gap
        for j in range(Gap):
            tmp = []
            for k in range(num_frames):
                tmp.append(List[start_id + j + k * Gap])
            ret.append(copy.deepcopy(tmp))

    if num * max_gap == len(List):
        return ret

    new_list = List[num * max_gap:]
    random.shuffle(new_list)
    ret.extend(np.array_split(new_list, len(new_list) // num_frames))
    return ret

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames = None, cache_mode=False, local_rank=0, local_size=1, is_train = False, \
                  gap=1, is_shuffle=False, filter_key_img=True, image_set=None):
        
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        
        self.image_set = image_set
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ann_file  = ann_file
        self.num_ref_frames = num_frames - 1
        self.cocovid = CocoVID(self.ann_file)
        self.is_train = is_train
        
        self.vid_ids = self.cocovid.get_vid_ids()
        self.filter_key_img = filter_key_img
        self.img_ids = []
        for vid_id in self.vid_ids:
            single_video_img_ids = self.cocovid.get_img_ids_from_vid(vid_id)
            while len(single_video_img_ids) < num_frames:
                single_video_img_ids.extend(copy.deepcopy(single_video_img_ids))
            nums = math.ceil(len(single_video_img_ids)* 1.0 / num_frames) # 4
            offset = nums * num_frames - len(single_video_img_ids) # 1
            if offset != 0 :
                single_video_img_ids.extend(copy.deepcopy(single_video_img_ids[-offset:]))
            if is_shuffle:
                random.shuffle(single_video_img_ids) 
            self.img_ids.extend(ChooseFrame(single_video_img_ids, gap, num_frames))

    def __len__(self):
        return len(self.img_ids) -1 
            
    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        imgs = [] 
        tgts = []
        #print("idx: ", idx)
        idxs = self.img_ids[idx]
        p1 = random.random()
        p2 = random.random()
        p = [p1, p2]
        # print("idxs: ", idxs)
        for i in idxs:    
            coco = self.coco
            img_id = self.ids[i-1]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            img_info = coco.loadImgs(img_id)[0]
            path = img_info['file_name']
            video_id = img_info['video_id']
            #print("i, video_id, img_id: ", i, video_id, img_id)
            img = self.get_image(path)
            target = {'image_id': img_id, 'video_id': video_id, 'annotations': target, 'path': path}
            img, target = self.prepare(img, target)
            imgs.append(img)
            tgts.append(target)

        imgs, target = self._transforms(imgs, tgts, p)
        
        return  torch.cat(imgs, dim=0),  target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        video_id = target['video_id']
        video_id = torch.tensor([video_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["video_id"] = video_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    if image_set == 'train_vid' or image_set == "train_det" or image_set == "train":
        return T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitterTransform(),
                T.RandomResize([640,640], max_size=640),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    if image_set == 'val' or 'valdn':
        return T.Compose([
                T.RandomResize([640,640], max_size=640),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    print(" In building the dataset: ", image_set)
    root = Path(args.vid_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": [('data/FL_Drones_full_train/Train/frames/', 'data/FL_Drones_full_train/Train/Train_annotations_all_frames.json')],
        "val": [('data/FL_Drones_full_train/Test/frames' , 'data/FL_Drones_full_train/Test/Test_annotations_every_4th_frame.json')]
            }
    datasets = []
    for (img_folder, ann_file) in PATHS[image_set]:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set=image_set), return_masks=args.masks, num_frames=args.num_frames, 
                                cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), is_train =(not args.eval), gap=args.gap, image_set=image_set)
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)