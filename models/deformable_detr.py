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
Deformable DETR model and criterion classes.
"""
import sys
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
					   accuracy, get_world_size, interpolate,
					   is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
						   dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as F_torchvision
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
	""" This is the Deformable DETR module that performs object detection """
	def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, time_steps,
				 aux_loss=True, with_box_refine=False, two_stage=False, use_dab=False, num_patterns=0,
				 random_refpoints_xy=False):
		""" Initializes the model.
		Parameters:
			backbone: torch module of the backbone to be used. See backbone.py
			transformer: torch module of the transformer architecture. See transformer.py
			num_classes: number of object classes
			num_queries: number of object queries, ie detection slot. This is the maximal number of objects
						 DETR can detect in a single image. For COCO, we recommend 100 queries.
			aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
			with_box_refine: iterative bounding box refinement
			two_stage: two-stage Deformable DETR
		"""
		super().__init__()
		self.num_queries = num_queries
		self.transformer = transformer
		hidden_dim = transformer.d_model
		self.class_embed = nn.Linear(hidden_dim, num_classes)
		self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
		self.num_feature_levels = num_feature_levels
		self.time_steps = time_steps
		#####################
		self.use_dab = use_dab
		self.num_patterns = num_patterns
		self.random_refpoints_xy = random_refpoints_xy
		######################
		if not two_stage:
			if not use_dab:  #######
				self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
			else: ###############
				self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
				self.refpoint_embed = nn.Embedding(num_queries, 4)
				if random_refpoints_xy:
					# import ipdb; ipdb.set_trace()
					self.refpoint_embed.weight.data[:, 2:].uniform_(0, 0.4) # change to 0.2 for NPS
					self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
					self.refpoint_embed.weight.data[:, :2].requires_grad = False
		
		
		if self.num_patterns > 0:  ##################
			self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)
   
		if num_feature_levels > 1:
			num_backbone_outs = len(backbone.strides)
			input_proj_list = []
			for _ in range(num_backbone_outs):
				in_channels = backbone.num_channels[_]
				input_proj_list.append(nn.Sequential(
					nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
					nn.GroupNorm(32, hidden_dim),
				))
			for _ in range(num_feature_levels - num_backbone_outs):
				input_proj_list.append(nn.Sequential(
					nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
					nn.GroupNorm(32, hidden_dim),
				))
				in_channels = hidden_dim
			self.input_proj = nn.ModuleList(input_proj_list)
		else:
			# self.input_proj = nn.ModuleList([
			#     nn.Sequential(
			#         nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
			#         nn.GroupNorm(32, hidden_dim),
			#     )])
			self.input_proj = nn.ModuleList([
				nn.Sequential(
					nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
					nn.GroupNorm(32, hidden_dim),
				)])
		self.backbone = backbone
		self.aux_loss = aux_loss
		self.with_box_refine = with_box_refine
		self.two_stage = two_stage

		prior_prob = 0.01
		bias_value = -math.log((1 - prior_prob) / prior_prob)
		self.class_embed.bias.data = torch.ones(num_classes) * bias_value
		nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
		nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
		for proj in self.input_proj:
			nn.init.xavier_uniform_(proj[0].weight, gain=1)
			nn.init.constant_(proj[0].bias, 0)

		# if two-stage, the last class_embed and bbox_embed is for region proposal generation
		num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
		if with_box_refine:
			self.class_embed = _get_clones(self.class_embed, num_pred)
			self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
			nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
			# hack implementation for iterative bounding box refinement
			self.transformer.decoder.bbox_embed = self.bbox_embed
		else:
			nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
			self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
			self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
			self.transformer.decoder.bbox_embed = None
		if two_stage:
			# hack implementation for two-stage
			self.transformer.decoder.class_embed = self.class_embed
			for box_embed in self.bbox_embed:
				nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
		
	def get_scaled(self,t,h=None,w=None,trans = True):
		if trans==True: t = self.transforms_internal(t,h,w)
		return (t-torch.min(t))/(torch.max(t)-torch.min(t))

	def forward(self, samples: NestedTensor, targets, epoch=None):
		""" The forward expects a NestedTensor, which consists of:
			   - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
			   - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

			It returns a dict with the following elements:
			   - "pred_logits": the classification logits (including no-object) for all queries.
								Shape= [batch_size x num_queries x (num_classes + 1)]
			   - "pred_boxes": The normalized boxes coordinates for all queries, represented as
							   (center_x, center_y, height, width). These values are normalized in [0, 1],
							   relative to the size of each individual image (disregarding possible padding).
							   See PostProcess for information on how to retrieve the unnormalized bounding box.
			   - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
								dictionnaries containing the two above keys for each decoder layer.
		"""
		if not isinstance(samples, NestedTensor):
			samples = nested_tensor_from_tensor_list(samples)
		features, object_enhanced_features, pos = self.backbone(samples)
		objectness_feats = torch.mean(object_enhanced_features, dim=1).to(object_enhanced_features.device)
		contour_coords_in_batch = []
		contour_xyr_in_batch = []
		for i in range(features[0].tensors.shape[0]):
			img_id = targets[i]['image_id']
			h, w = int(targets[i]['size'][0]), int(targets[i]['size'][1])
			# objectness_mask =  F_torchvision.resize( (1-self.get_scaled(objectness_feats[i], trans=False)).unsqueeze(0), (h,w) ).sigmoid()
			objectness_mask = F.interpolate((1 - self.get_scaled(objectness_feats[i], trans=False)).unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).sigmoid()
			objectness_mask = ((objectness_mask>0.6)*255.0).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
			contours, hierarchy = cv2.findContours(objectness_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			for component in zip(contours, hierarchy[0]):
				currentContour = component[0]
				currentHierarchy = component[1]
				contours_poly = cv2.approxPolyDP(currentContour, 3, True)
				center, radius = cv2.minEnclosingCircle(contours_poly)
				x1, y1, w1, h1 = cv2.boundingRect(contours_poly)
				if currentHierarchy[3] < 0: # these are the outermost parent components
					contour_coords_in_batch.append([int(x1), int(y1), int(w1), int(h1)])
					contour_xyr_in_batch.append(np.array([int(center[0]), int(center[1]), int(radius)]))
		contour_xyr_in_batch_repeated = []
		for i in range(features[0].tensors.shape[0]):
			contour_xyr_in_batch_repeated.append(np.array(contour_xyr_in_batch))
	
		srcs = []
		masks = []
		for l, feat in enumerate(features):
			src, mask = feat.decompose()
			srcs.append(self.input_proj[l](src))
			masks.append(mask)
			assert mask is not None
		if self.num_feature_levels > len(srcs):
			_len_srcs = len(srcs)
			for l in range(_len_srcs, self.num_feature_levels):
				if l == _len_srcs:
					src = self.input_proj[l](features[-1].tensors)
				else:
					src = self.input_proj[l](srcs[-1])
				m = samples.mask
				mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
				pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
				srcs.append(src)
				masks.append(mask)
				pos.append(pos_l)

		if self.two_stage:
			query_embeds = None
		elif self.use_dab:
			if self.num_patterns == 0:
				tgt_embed = self.tgt_embed.weight           # nq, 256
				refanchor = self.refpoint_embed.weight      # nq, 4
				query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
			else:
				# multi patterns
				tgt_embed = self.tgt_embed.weight           # nq, 256
				pat_embed = self.patterns_embed.weight      # num_pat, 256
				tgt_embed = tgt_embed.repeat(self.num_patterns, 1) # nq*num_pat, 256
				pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1) # nq*num_pat, 256
				tgt_all_embed = tgt_embed + pat_embed
				refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)      # nq*num_pat, 4
				query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
		else:
			query_embeds = self.query_embed.weight
		
		hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory, init_ref_out = self.transformer(srcs, masks, pos, query_embeds, targets, contour_coords_in_batch, epoch=epoch)	
		
		outputs_classes = []
		outputs_coords = []
		for lvl in range(hs.shape[0]):
			if lvl == 0:
				reference = init_reference
			else:
				reference = inter_references[lvl - 1]
			reference = inverse_sigmoid(reference)
			outputs_class = self.class_embed[lvl](hs[lvl])
			tmp = self.bbox_embed[lvl](hs[lvl])
			if reference.shape[-1] == 4:
				tmp += reference
			else:
				assert reference.shape[-1] == 2
				tmp[..., :2] += reference
			outputs_coord = tmp.sigmoid()
			outputs_classes.append(outputs_class)
			outputs_coords.append(outputs_coord)
		outputs_class = torch.stack(outputs_classes)
		outputs_coord = torch.stack(outputs_coords)

		out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
		if self.aux_loss:
			out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

		if self.two_stage:
			enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
			out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
		# return out
		return out, objectness_feats, contour_xyr_in_batch_repeated, inter_references

	@torch.jit.unused
	def _set_aux_loss(self, outputs_class, outputs_coord):
		# this is a workaround to make torchscript happy, as torchscript
		# doesn't support dictionary with non-homogeneous values, such
		# as a dict having both a Tensor and a list.
		return [{'pred_logits': a, 'pred_boxes': b}
				for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
	""" This class computes the loss for DETR.
	The process happens in two steps:
		1) we compute hungarian assignment between ground truth boxes and the outputs of the model
		2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
	"""
	def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
		""" Create the criterion.
		Parameters:
			num_classes: number of object categories, omitting the special no-object category
			matcher: module able to compute a matching between targets and proposals
			weight_dict: dict containing as key the names of the losses and as values their relative weight.
			losses: list of all the losses to be applied. See get_loss for list of available losses.
			focal_alpha: alpha in Focal Loss
		"""
		super().__init__()
		self.num_classes = num_classes
		self.matcher = matcher
		self.weight_dict = weight_dict
		self.losses = losses
		self.focal_alpha = focal_alpha

	def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
		"""Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""
		assert 'pred_logits' in outputs
		src_logits = outputs['pred_logits']

		idx = self._get_src_permutation_idx(indices)
		target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
		target_classes = torch.full(src_logits.shape[:2], self.num_classes,
									dtype=torch.int64, device=src_logits.device)
		target_classes[idx] = target_classes_o

		target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
											dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
		target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

		target_classes_onehot = target_classes_onehot[:,:,:-1]
		loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
		losses = {'loss_ce': loss_ce}

		if log:
			# TODO this should probably be a separate loss, not hacked in this one here
			losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
		return losses

	@torch.no_grad()
	def loss_cardinality(self, outputs, targets, indices, num_boxes):
		""" Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
		This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
		"""
		pred_logits = outputs['pred_logits']
		device = pred_logits.device
		tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
		# Count the number of predictions that are NOT "no-object" (which is the last class)
		card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
		card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
		losses = {'cardinality_error': card_err}
		return losses

	def loss_boxes(self, outputs, targets, indices, num_boxes):
		"""Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
		   targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
		   The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
		"""
		assert 'pred_boxes' in outputs
		idx = self._get_src_permutation_idx(indices)
		src_boxes = outputs['pred_boxes'][idx]
		target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

		loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

		losses = {}
		losses['loss_bbox'] = loss_bbox.sum() / num_boxes

		loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
			box_ops.box_cxcywh_to_xyxy(src_boxes),
			box_ops.box_cxcywh_to_xyxy(target_boxes)))
		losses['loss_giou'] = loss_giou.sum() / num_boxes
		return losses

	def loss_masks(self, outputs, targets, indices, num_boxes):
		"""Compute the losses related to the masks: the focal loss and the dice loss.
		   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
		"""
		assert "pred_masks" in outputs

		src_idx = self._get_src_permutation_idx(indices)
		tgt_idx = self._get_tgt_permutation_idx(indices)

		src_masks = outputs["pred_masks"]

		# TODO use valid to mask invalid areas due to padding in loss
		target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
		target_masks = target_masks.to(src_masks)

		src_masks = src_masks[src_idx]
		# upsample predictions to the target size
		src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
								mode="bilinear", align_corners=False)
		src_masks = src_masks[:, 0].flatten(1)

		target_masks = target_masks[tgt_idx].flatten(1)

		losses = {
			"loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
			"loss_dice": dice_loss(src_masks, target_masks, num_boxes),
		}
		return losses

	def dice_loss_(self, input, target):
		smooth = 1e-5  # small constant to avoid division by zero
		input_flat = input.view(-1)
		target_flat = target.view(-1)
		intersection = (input_flat * target_flat).sum()
		dice_coeff = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
		loss = 1. - dice_coeff
		return loss

	def feat_aggr_attn_loss(self, attn_maps, gt_maps):
		predicted_mask = attn_maps.view(-1).sigmoid()
		ground_truth_mask = gt_maps.view(-1)
		losses = { "feat_aggr_attn_loss": self.dice_loss_(predicted_mask, ground_truth_mask) / attn_maps.shape[0] }
		return losses
	
	def instance_aware_feat_aggr_attn_loss(self, attn_maps, gt_maps, instance_wise_gt_maps):
		predicted_mask = attn_maps.view(-1).sigmoid()
		ground_truth_mask = gt_maps.view(-1)
		# Compute the instance-aware loss
		loss = 0
		num_instances = torch.unique(instance_wise_gt_maps).numel()
		if num_instances>1:
			for instance_id in range(1, num_instances):
				mask = (instance_wise_gt_maps == instance_id).float().view(-1)
				predicted_mask_instance = predicted_mask * mask
				ground_truth_mask_instance = ground_truth_mask * mask
				# Compute the binary cross entropy loss for the current instance
				loss_instance = nn.functional.binary_cross_entropy(predicted_mask_instance, ground_truth_mask_instance)
				loss += loss_instance
		else:
			print("Unique instances are LTE 1")
			loss = torch.tensor(loss)
		loss /= num_instances
		return {'instance_aware_feat_aggr_attn_loss': loss}

	def dab_detr_decoder_query_loss(self, contour_list, inter_references):
		'''
		contour : (bs, num_contours, cx, cy, r)
		inter_references : (num_decoder_layers=6, bs, num_queries=100, 4).
		'''
		loss = 0
		if len(contour_list)==0 or contour_list is None: 
			print("--- NO CONTOURS ---")
			return {'dab_detr_decoder_query_loss':torch.tensor(loss).float().to(inter_references.device)}

		_, bs, num_q, _ = inter_references.shape
		rect_coords = []
		for i in range(bs):
			rect_coords_per_img = []
			queries_layer_1 = inter_references[-1,i,:,:]
			contours = contour_list[i]

			dist_1 = torch.cdist(queries_layer_1[:,:2].float(),torch.tensor(contours[:,:2]).float().to(inter_references.device),p=2.0)		
			radius = contours[:,2]
			diff_1 = dist_1 - torch.tensor(radius).to(inter_references.device)
			
			# loss += torch.sum(torch.max(diff*(diff>0),dim=0).values) # for every contour, find farthest point and add dist
			loss += torch.sum(torch.min(diff_1*(diff_1>0),dim=1).values) # for every point, find closest contour and add dist

		return {'dab_detr_decoder_query_loss':(loss/(bs)).to(inter_references.device)}

	def dab_detr_decoder_query_width_and_height_loss(self, inter_references):
		'''
		inter_references : (num_decoder_layers=6, bs, num_queries=100, 4).
		'''
		loss = 0
		for i in range(inter_references.shape[1]):
			queries = inter_references[-1,i,:,2:]
			queries = queries - 0.4 # 0.2 for NPS Drones dataset
			loss += torch.sum(queries[queries>0])
		return {'dab_detr_decoder_query_width_and_height_loss':(loss/(inter_references.shape[1])).to(inter_references.device)}
 
	def _get_src_permutation_idx(self, indices):
		# permute predictions following indices
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def _get_tgt_permutation_idx(self, indices):
		# permute targets following indices
		batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
		tgt_idx = torch.cat([tgt for (_, tgt) in indices])
		return batch_idx, tgt_idx

	def get_loss(self, loss, outputs, targets, indices, num_boxes, attn_maps=None, gt_maps=None, instance_wise_gt_maps=None, contour_list=None, inter_references=None, **kwargs):
		loss_map = {
			'labels': self.loss_labels,
			'cardinality': self.loss_cardinality,
			'boxes': self.loss_boxes,
			'masks': self.loss_masks,
			'feat_aggr_attn_loss': self.feat_aggr_attn_loss,
			'instance_aware_feat_aggr_attn_loss': self.instance_aware_feat_aggr_attn_loss,
			'dab_detr_decoder_query_loss': self.dab_detr_decoder_query_loss,
			'dab_detr_decoder_query_width_and_height_loss': self.dab_detr_decoder_query_width_and_height_loss
		}
		assert loss in loss_map, f'do you really want to compute {loss} loss?'
		if loss == 'feat_aggr_attn_loss': return loss_map[loss](attn_maps, gt_maps)
		if loss == 'instance_aware_feat_aggr_attn_loss': return loss_map[loss](attn_maps, gt_maps, instance_wise_gt_maps)
		if loss == 'dab_detr_decoder_query_loss': return loss_map[loss](contour_list, inter_references)
		if loss == 'dab_detr_decoder_query_width_and_height_loss': return loss_map[loss](inter_references)
		return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

	def get_scaled(self,t,h=None,w=None,trans = True):
		if trans==True: t = self.transforms_internal(t,h,w)
		return (t-torch.min(t))/(torch.max(t)-torch.min(t))

	def forward(self, outputs, targets, samples=None, attn_maps_orig=None, contour_list=None, inter_references=None, epoch=None):
		""" This performs the loss computation.
		Parameters:
			 outputs: dict of tensors, see the output specification of the model for the format
			 targets: list of dicts, such that len(targets) == batch_size.
					  The expected keys in each dict depends on the losses applied, see each loss' doc
		"""
		outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
		
		device = samples.tensors.device
		attn_maps = torch.empty(samples.tensors[:,0,:,:].shape, requires_grad=True).unsqueeze(1).to(device)
		gt_maps = torch.zeros(attn_maps.shape).to(device)
		instance_wise_gt_maps = torch.zeros(attn_maps.shape).to(device) # (bs, 1, h, w)
		for i in range(len(targets)):
			h, w = int(targets[i]['size'][0]), int(targets[i]['size'][1])
			attn_maps[i] = F.interpolate((1 - self.get_scaled(attn_maps_orig[i], trans=False)).unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
			# attn_maps[i] = self.get_scaled(attn_maps_orig[i], h, w, trans=True) #.unsqueeze(0)
			for j in range(len(targets[i]['boxes'])):
				start_x, start_y, end_x, end_y = box_cxcywh_to_xyxy(targets[i]['boxes'][j] * torch.tensor([w,h,w,h], dtype=torch.float32).to(device)).to(device)
				gt_maps[i][0][int(start_y):int(end_y),int(start_x):int(end_x)] = 1
				instance_wise_gt_maps[i][0][int(start_y):int(end_y),int(start_x):int(end_x)] = j+1
				
		# Retrieve the matching between the outputs of the last layer and the targets
		indices = self.matcher(outputs_without_aux, targets)

		# Compute the average number of target boxes accross all nodes, for normalization purposes
		num_boxes = sum(len(t["labels"]) for t in targets)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
		if is_dist_avail_and_initialized():
			torch.distributed.all_reduce(num_boxes)
		num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

		# Compute all the requested losses
		losses = {}
		for loss in self.losses:
			kwargs = {}
			if loss=='feat_aggr_attn_loss': losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, attn_maps,  gt_maps, **kwargs))
			elif loss=='instance_aware_feat_aggr_attn_loss': losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, attn_maps,  gt_maps, instance_wise_gt_maps, **kwargs))
			elif loss=='dab_detr_decoder_query_loss': 
				l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, contour_list=contour_list, inter_references=inter_references,**kwargs)
				l_dict['dab_detr_decoder_query_loss'] = l_dict['dab_detr_decoder_query_loss']* float(epoch>0)
				losses.update(l_dict)
			elif loss=='dab_detr_decoder_query_width_and_height_loss': losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, inter_references=inter_references, **kwargs))
			else: losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, attn_maps=None, gt_maps=None, contour_list=None, inter_references=None, **kwargs))

		# In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
		if 'aux_outputs' in outputs:
			for i, aux_outputs in enumerate(outputs['aux_outputs']):
				indices = self.matcher(aux_outputs, targets)
				for loss in self.losses:
					if loss == 'masks':
						# Intermediate masks losses are too costly to compute, we ignore them.
						continue
					kwargs = {}
					if loss == 'labels':
						# Logging is enabled only for the last layer
						kwargs['log'] = False
					if loss=='feat_aggr_attn_loss': l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, attn_maps, gt_maps, **kwargs)
					elif loss == 'instance_aware_feat_aggr_attn_loss': l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, attn_maps, gt_maps, instance_wise_gt_maps, **kwargs)
					elif loss=='dab_detr_decoder_query_loss': 
						l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, contour_list=contour_list, inter_references=inter_references,**kwargs)
						l_dict['dab_detr_decoder_query_loss'] = l_dict['dab_detr_decoder_query_loss']* float(epoch>0)
					elif loss=='dab_detr_decoder_query_width_and_height_loss': l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, inter_references=inter_references, **kwargs)
					else: l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, attn_maps=None, gt_maps=None, **kwargs)
					l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
					losses.update(l_dict)

		if 'enc_outputs' in outputs:
			enc_outputs = outputs['enc_outputs']
			bin_targets = copy.deepcopy(targets)
			for bt in bin_targets:
				bt['labels'] = torch.zeros_like(bt['labels'])
			indices = self.matcher(enc_outputs, bin_targets)
			for loss in self.losses:
				if loss == 'masks':
					# Intermediate masks losses are too costly to compute, we ignore them.
					continue
				kwargs = {}
				if loss == 'labels':
					# Logging is enabled only for the last layer
					kwargs['log'] = False
				if loss=='feat_aggr_attn_loss': l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, attn_maps, gt_maps, **kwargs)
				elif loss == 'instance_aware_feat_aggr_attn_loss': l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, attn_maps, gt_maps, instance_wise_gt_maps, **kwargs)
				elif loss=='dab_detr_decoder_query_loss': 
					l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, contour_list=contour_list, inter_references=inter_references,**kwargs)
					l_dict['dab_detr_decoder_query_loss'] = l_dict['dab_detr_decoder_query_loss']* float(epoch>0)
				elif loss=='dab_detr_decoder_query_width_and_height_loss': l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, inter_references=inter_references, **kwargs)
				else: l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, attn_maps=None, gt_maps=None, **kwargs)
				l_dict = {k + f'_enc': v for k, v in l_dict.items()}
				losses.update(l_dict)

		return losses


class PostProcess(nn.Module):
	""" This module converts the model's output into the format expected by the coco api"""

	@torch.no_grad()
	def forward(self, outputs, target_sizes):
		""" Perform the computation
		Parameters:
			outputs: raw outputs of the model
			target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
						  For evaluation, this must be the original image size (before any data augmentation)
						  For visualization, this should be the image size after data augment, but before padding
		"""
		out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

		assert len(out_logits) == len(target_sizes)
		assert target_sizes.shape[1] == 2

		prob = out_logits.sigmoid()
		topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
		scores = topk_values
		topk_boxes = topk_indexes // out_logits.shape[2]
		labels = topk_indexes % out_logits.shape[2]
		boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
		boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

		# and from relative [0, 1] to absolute [0, height] coordinates
		img_h, img_w = target_sizes.unbind(1)
		scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
		boxes = boxes * scale_fct[:, None, :]

		results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

		return results


class MLP(nn.Module):
	""" Very simple multi-layer perceptron (also called FFN)"""

	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
		return x


def build(args):
	num_classes = 2
	device = torch.device(args.device)

	if 'swin' in args.backbone:
		from .swin_transformer import build_swin_backbone
		backbone = build_swin_backbone(args) 
	else:
		backbone = build_backbone(args)
	# backbone = build_backbone(args)

	transformer = build_deforamble_transformer(args)
	model = DeformableDETR(
		backbone,
		transformer,
		num_classes=num_classes,
		num_queries=args.num_queries,
		num_feature_levels=args.num_feature_levels,
		aux_loss=args.aux_loss,
		with_box_refine=args.with_box_refine,
		two_stage=args.two_stage,
		time_steps = args.num_frames,
		use_dab=args.use_dab,
		num_patterns=args.num_patterns,
		random_refpoints_xy=args.random_refpoints_xy
	)
	if args.masks:
		model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
	matcher = build_matcher(args)
	weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
	weight_dict['loss_giou'] = args.giou_loss_coef
	weight_dict['feat_aggr_attn_loss'] = args.feat_aggr_attn_loss_coef
	weight_dict['instance_aware_feat_aggr_attn_loss'] = args.instance_aware_feat_aggr_attn_loss_coef
	weight_dict['dab_detr_decoder_query_loss'] = args.dab_detr_decoder_query_loss_coef
	weight_dict['dab_detr_decoder_query_width_and_height_loss'] = args.dab_detr_decoder_query_width_and_height_loss_coef
	
	if args.masks:
		weight_dict["loss_mask"] = args.mask_loss_coef
		weight_dict["loss_dice"] = args.dice_loss_coef
	# TODO this is a hack
	if args.aux_loss:
		aux_weight_dict = {}
		for i in range(args.dec_layers - 1):
			aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
		aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
		weight_dict.update(aux_weight_dict)

	losses = ['labels', 'boxes', 'cardinality', 'feat_aggr_attn_loss', 'instance_aware_feat_aggr_attn_loss', \
				'dab_detr_decoder_query_loss', 'dab_detr_decoder_query_width_and_height_loss']
	if args.masks:
		losses += ["masks"]
	# num_classes, matcher, weight_dict, losses, focal_alpha=0.25
	criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
	criterion.to(device)
	postprocessors = {'bbox': PostProcess()}
	if args.masks:
		postprocessors['segm'] = PostProcessSegm()
		if args.dataset_file == "coco_panoptic":
			is_thing_map = {i: i <= 90 for i in range(201)}
			postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

	return model, criterion, postprocessors
