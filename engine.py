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
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import time
import torch
from torch.autograd import grad
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.box_ops import box_cxcywh_to_xyxy
import cv2
import collections
import sys

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
					data_loader: Iterable, optimizer: torch.optim.Optimizer,
					device: torch.device, epoch: int, max_norm: float = 0, classifier=None):
	model.train()
	criterion.train()
	metric_logger = utils.MetricLogger(delimiter="  ")
	metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
	metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
	metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
	header = 'Epoch: [{}]'.format(epoch)
	print("------------------------------------------------------!!!!")
	for samples, targets in metric_logger.log_every(data_loader, 10, header):       

		samples = samples.to(device)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets[0]]
		samples.tensors = samples.tensors.view(int(samples.tensors.shape[1]/3), 3, int(samples.tensors.shape[2]), int(samples.tensors.shape[3]))

		outputs, feat_aggr_attn_maps, contour_list, inter_references = model(samples, targets, epoch=epoch+1) 
		loss_dict = criterion(outputs, targets, samples, attn_maps_orig=feat_aggr_attn_maps, contour_list=contour_list, inter_references=inter_references, epoch=epoch+1)

		weight_dict = criterion.weight_dict
		losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

		# reduce losses over all GPUs for logging purposes
		loss_dict_reduced = utils.reduce_dict(loss_dict)
		loss_dict_reduced_unscaled = {f'{k}_unscaled': v
									for k, v in loss_dict_reduced.items()}
		loss_dict_reduced_scaled = {k: v * weight_dict[k]
									for k, v in loss_dict_reduced.items() if k in weight_dict}
		losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

		loss_value = losses_reduced_scaled.item()

		if not math.isfinite(loss_value):
			print("Loss is {}, stopping training".format(loss_value))
			print(loss_dict_reduced)
			sys.exit(1)

		optimizer.zero_grad()
		losses.backward()

		if max_norm > 0:
			grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
		else:
			grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
		optimizer.step()

		metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
		metric_logger.update(class_error=loss_dict_reduced['class_error'])
		metric_logger.update(lr=optimizer.param_groups[0]["lr"])
		metric_logger.update(grad_norm=grad_total_norm)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)        
	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}#,losses

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir): # classifier_bb, classifier_ll
	#half = True 
	#half &= device.type != 'cpu'  # half precision only supported on CUDA
	#model.half() if half else model.float()
	# model = model.half()
	model.eval()
	criterion.eval()

	metric_logger = utils.MetricLogger(delimiter="  ")
	metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
	header = 'Test:'

	iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
	coco_evaluator = CocoEvaluator(base_ds, iou_types)
	# coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

	panoptic_evaluator = None
	if 'panoptic' in postprocessors.keys():
		panoptic_evaluator = PanopticEvaluator(
			data_loader.dataset.ann_file,
			data_loader.dataset.ann_folder,
			output_dir=os.path.join(output_dir, "panoptic_eval"),
		)

	for samples, targets in metric_logger.log_every(data_loader, 10, header):
		samples = samples.to(device)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets[0]]
		samples.tensors = samples.tensors.view(int(samples.tensors.shape[1]/3), 3, int(samples.tensors.shape[2]), int(samples.tensors.shape[3])) 
  
		outputs, feat_aggr_attn_maps, contour_list, inter_references = model(samples, targets, epoch=15) 
		loss_dict = criterion(outputs, targets, samples, attn_maps_orig=feat_aggr_attn_maps, contour_list=contour_list, inter_references=inter_references, epoch=15) 

		weight_dict = criterion.weight_dict

		# reduce losses over all GPUs for logging purposes
		loss_dict_reduced = utils.reduce_dict(loss_dict)
		loss_dict_reduced_scaled = {k: v * weight_dict[k]
									for k, v in loss_dict_reduced.items() if k in weight_dict}
		loss_dict_reduced_unscaled = {f'{k}_unscaled': v
									for k, v in loss_dict_reduced.items()}
		metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
							**loss_dict_reduced_scaled,
							**loss_dict_reduced_unscaled)
		metric_logger.update(class_error=loss_dict_reduced['class_error'])

		orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
		results = postprocessors['bbox'](outputs, orig_target_sizes)

		if 'segm' in postprocessors.keys():
			target_sizes = torch.stack([t["size"] for t in targets], dim=0)
			results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
		res = {target['image_id'].item(): output for target, output in zip(targets, results)}
		if coco_evaluator is not None:
			coco_evaluator.update(res)

		if panoptic_evaluator is not None:
			res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
			for i, target in enumerate(targets):
				image_id = target["image_id"].item()
				file_name = f"{image_id:012d}.png"
				res_pano[i]["image_id"] = image_id
				res_pano[i]["file_name"] = file_name

			panoptic_evaluator.update(res_pano)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	if coco_evaluator is not None:
		coco_evaluator.synchronize_between_processes()
	if panoptic_evaluator is not None:
		panoptic_evaluator.synchronize_between_processes()

	# accumulate predictions from all images
	if coco_evaluator is not None:
		coco_evaluator.accumulate()
		coco_evaluator.summarize()
	panoptic_res = None
	if panoptic_evaluator is not None:
		panoptic_res = panoptic_evaluator.summarize()
	stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
	if coco_evaluator is not None:
		if 'bbox' in postprocessors.keys():
			stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
		if 'segm' in postprocessors.keys():
			stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
	if panoptic_res is not None:
		stats['PQ_all'] = panoptic_res["All"]
		stats['PQ_th'] = panoptic_res["Things"]
		stats['PQ_st'] = panoptic_res["Stuff"]
	return stats, coco_evaluator


@torch.no_grad()
def evaluate_fps(model, data_loader, device): # classifier_bb, classifier_ll
	#half = True 
	#half &= device.type != 'cpu'  # half precision only supported on CUDA
	#model.half() if half else model.float()
	model.eval()

	total_time = 0
	total_imgs = 0
	for i, (samples, targets) in enumerate(data_loader):
		samples = samples.to(device)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets[0]]
		samples.tensors = samples.tensors.view(int(samples.tensors.shape[1]/3), 3, int(samples.tensors.shape[2]), int(samples.tensors.shape[3]))
		
		start = time.time()
		outputs, feat_aggr_attn_maps, contour_list, inter_references = model(samples, targets, epoch=15)
		total_time +=  (time.time() - start)
		total_imgs += len(samples.tensors)

	print("len Samples: ", total_imgs)
	return total_imgs/total_time

@torch.no_grad()
def validate(model, criterion, postprocessors, data_loader, base_ds, device, epoch, output_dir):

	model.eval()
	criterion.eval()
	
	metric_logger = utils.MetricLogger(delimiter="  ")
	metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
	header = 'VALDN:'

	iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
	coco_evaluator = CocoEvaluator(base_ds, iou_types)

	for samples, targets in metric_logger.log_every(data_loader, 10, header):
		samples = samples.to(device)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets[0]]
		samples.tensors = samples.tensors.view(int(samples.tensors.shape[1]/3), 3, int(samples.tensors.shape[2]), int(samples.tensors.shape[3]))
	
		outputs, feat_aggr_attn_maps, contour_list, inter_references = model(samples, targets, epoch=epoch+1) 
		loss_dict = criterion(outputs, targets, samples, attn_maps_orig=feat_aggr_attn_maps, contour_list=contour_list, inter_references=inter_references, epoch=epoch+1) 
		
		weight_dict = criterion.weight_dict

		# reduce losses over all GPUs for logging purposes
		loss_dict_reduced = utils.reduce_dict(loss_dict)
		loss_dict_reduced_scaled = {k: v * weight_dict[k]
									for k, v in loss_dict_reduced.items() if k in weight_dict}
		loss_dict_reduced_unscaled = {f'{k}_unscaled': v
									for k, v in loss_dict_reduced.items()}
		metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
							**loss_dict_reduced_scaled,
							**loss_dict_reduced_unscaled)
		metric_logger.update(class_error=loss_dict_reduced['class_error'])

		
		orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
		results = postprocessors['bbox'](outputs, orig_target_sizes)

		res = {target['image_id'].item(): output for target, output in zip(targets, results)}
		if coco_evaluator is not None:
			coco_evaluator.update(res)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	if coco_evaluator is not None:
		coco_evaluator.synchronize_between_processes()

	# accumulate predictions from all images
	if coco_evaluator is not None:
		coco_evaluator.accumulate()
		coco_evaluator.summarize()
	
	stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
	if coco_evaluator is not None:
		if 'bbox' in postprocessors.keys():
			stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
		
	return stats, coco_evaluator
