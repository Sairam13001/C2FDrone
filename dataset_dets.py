# 
import os
import sys
import json
import shutil


def get_all_splits(w,h,num_splits):
	out = set()
	start_x, start_y,end_x,end_y = 0,0,w//num_splits,h//num_splits
	for i in range(1,num_splits**2+1):
		out.add((start_x//8, start_y//8,end_x//8,end_y//8))
		start_x = (i%num_splits)*(w//num_splits)
		start_y = (i//num_splits)*(h//num_splits)
		end_x = w//num_splits + (i%num_splits)*(w//num_splits)
		end_y = h//num_splits + (i//num_splits)*(h//num_splits)
			
	return out

dataset_name = 'FL_full_train'
# if dataset_name in ['NPS_val', 'NPS_val_sample']: VALIDATION_FLAG = True
# else: VALIDATION_FLAG = False

VALIDATION_FLAG = False

dataset_details ={  'FL': {'root_path': 'data/FL_Drones_half_split/',
						'train_frames_path': 'data/FL_Drones_half_split/train/frames/', 
						'test_frames_path': 'data/FL_Drones_half_split/test/frames/', 
						'train_ann_path':'data/FL_Drones_half_split/train/train_annotations_4_transvod_lite_vid_hs.json', 
						'test_ann_path':'data/FL_Drones_half_split/test/test_annotations_4_transvod_lite_hs.json', 
						'res':[752,480]},
					'FL_val': {'root_path': 'data/FL_Drones_half_split_val/',
						'train_frames_path': 'data/FL_Drones_half_split_val/train/frames/', 
                        'val_frames_path': 'data/FL_Drones_half_split_val/val/frames/',
						'test_frames_path': 'data/FL_Drones_half_split_val/test/frames/', 
						'train_ann_path':'data/FL_Drones_half_split_val/train/train_annotations_4_transvod_lite_vid_with_val.json',
                        'val_ann_path':'data/FL_Drones_half_split_val/val/val_annotations_4_transvod_lite_vid.json', 
						'test_ann_path':'data/FL_Drones_half_split_val/test/test_annotations_4_transvod_lite.json', 
						'res':[640,640]},
                    'FL_full_train': {'root_path': 'data/FL_Drones_full_train/',
						'train_frames_path': 'data/FL_Drones_full_train/Train/frames/', 
						'test_frames_path': 'data/FL_Drones_full_train/Test/frames/', 
						'train_ann_path':'data/FL_Drones_full_train/Train/Train_annotations_all_frames.json',
						'test_ann_path':'data/FL_Drones_full_train/Test/Test_annotations_every_4th_frame.json', 
						'res':[1920,1280]},
					'NPS': {'root_path': 'data/NPS_Drones/',
						'train_frames_path': 'data/NPS_Drones/train/frames/', 
						'test_frames_path': 'data/NPS_Drones/test/frames/', 
						'train_ann_path':'data/NPS_Drones/train/train_annotations_4_transvod_lite_vid.json', 
						'test_ann_path':'data/NPS_Drones/test/test_annotations_4_transvod_lite.json', 
						'res':[640,640]},
					'NPS_val': {'root_path': 'data/NPS_Drones_val/',
						'train_frames_path': 'data/NPS_Drones_val/train/frames_inpainted_full_size/', 
						'val_frames_path': 'data/NPS_Drones_val/val/frames/', 
						'test_frames_path': 'data/NPS_Drones_val/test/frames/', 
						'train_ann_path':'data/NPS_Drones_val/train/train_annotations_4_transvod_lite_vid_with_val.json',
						'val_ann_path':'data/NPS_Drones_val/val/val_annotations_4_transvod_lite_vid.json', 
						'test_ann_path':'data/NPS_Drones_val/test/test_annotations_4_transvod_lite.json', 
						'res':[640, 640]},
                    'NPS_val_sample': {'root_path': 'data/NPS_Drones_val_sample/',
						'train_frames_path': 'data/NPS_Drones_val_sample/train/frames/', 
						'val_frames_path': 'data/NPS_Drones_val_sample/val/frames/', 
						'test_frames_path': 'data/NPS_Drones_val_sample/test/frames/', 
						'train_ann_path':'data/NPS_Drones_val_sample/train/train_annotations_4_transvod_lite_vid_with_val_sample.json',
						'val_ann_path':'data/NPS_Drones_val_sample/val/val_annotations_4_transvod_lite_vid_sample.json', 
						'test_ann_path':'data/NPS_Drones_val_sample/test/test_annotations_4_transvod_lite_sample.json', 
						'res':[640, 640]}
				}
#############
TEMP_DATA_LOADER_FLAG = True
DAB_DETR_FLAG = True
##############
K=6

DENSE_AGGREGATED_FEATS = True
dense_aggrgation_styles = {'1': 'fpn_style_da', '2': 'da_one_layer', '3': 'da_orig', '4': 'fpn_style_da_plus_fpn'}
if DENSE_AGGREGATED_FEATS: DA_STYLE = dense_aggrgation_styles['4']
else: DA_STYLE = None

ATTN_MAP_LOSS = True
INSTANCE_AWARE_ATTN_MAP_LOSS = True
INITIALIZE_DECODER_QUERIES = True
INIT_USING_ATTN_MAPS=True; INIT_USING_GTS = not INIT_USING_ATTN_MAPS; INIT_PER_IMG = False; INIT_PER_BATCH = not INIT_PER_IMG
INIT_DEC_Q_WH = False; WH_MAX = 0.20 if 'NPS' in dataset_name else 0.4 # Check the range -> Do you want it to be (0 to 0.4) or something else?
DECODER_QUERY_LOSS = True
if DECODER_QUERY_LOSS: LAYER_L_DECODER_QUERY_LOSS_ALONG_WITH_LAYER_1 = False; LAYER_L=4; AVG=False
RECT_CONT_LOSS = False
DECODER_QUERY_WH_LOSS = True
if DECODER_QUERY_WH_LOSS: ALL_LAYERS=False; ONLY_LAST_LAYER = not ALL_LAYERS

FEATURE_MASKING_AROUND_CONTOURS=False; FEATURE_MASKING_SPLITS=False; MASK_USING_ATTN_MAPS=True; MASK_USING_GT = not MASK_USING_ATTN_MAPS

VISUALISE_MEAN_FEAT_MAPS = False
VISUALISE_DAB_DETR_QUERIES = False

THREE_ENC_DEC_LAYERS = False


if VISUALISE_MEAN_FEAT_MAPS:
	ATTN_MAPS_FOLDER = 'Attn_maps_trials/'
	if os.path.exists(ATTN_MAPS_FOLDER) and len(os.listdir(ATTN_MAPS_FOLDER))>1: shutil.rmtree(ATTN_MAPS_FOLDER)
	os.makedirs(ATTN_MAPS_FOLDER, exist_ok=True)
        
if VISUALISE_DAB_DETR_QUERIES:
	DEC_QUERIES_FOLDER  = 'Init_refs_trials/'
	if os.path.exists(DEC_QUERIES_FOLDER) and len(os.listdir(DEC_QUERIES_FOLDER))>1: shutil.rmtree(DEC_QUERIES_FOLDER)
	os.makedirs(DEC_QUERIES_FOLDER, exist_ok=True)



########################################################
if DECODER_QUERY_LOSS == True and DENSE_AGGREGATED_FEATS == False:
	print(" !! DECODER QUERY LOSS NEEDS DENSE_AGGREGATED_FEATS !! CHECK THE FLAGS AGAIN")
	sys.exit(0)
        
if DECODER_QUERY_LOSS and DENSE_AGGREGATED_FEATS:
    print(" --- Both Attn map and Decoder query Losses are operating --- ")
    
elif DENSE_AGGREGATED_FEATS and ATTN_MAP_LOSS:
    print(" --- Only Atttn map loss operating --- ")
    
elif not ATTN_MAP_LOSS and not DECODER_QUERY_LOSS:
    print("---- None of the attn map loss or decoder query loss is operating -----")

#######################################################
CLASSIFIER_FLAG = False
SWIN_B_P4W7_CLASSIFIER = False
IMG_SPLITTING_AND_FEATURE_STITCHING_FLAG = False
FEAT_MASKING_FLAG = False
NUM_SPLITS=3
ALL_SPLITS_SET = get_all_splits(dataset_details[dataset_name]['res'][0],dataset_details[dataset_name]['res'][1],NUM_SPLITS)




VISUALIZE = False # set true for visualizing anything where we donot want the augmentations
TRAIN_STATUS = False if '--eval' in sys.argv else True
if TRAIN_STATUS==False:
    with open(dataset_details[dataset_name]['test_ann_path'], 'r')  as f:
        vis_data = json.load(f)
else:
    with open(dataset_details[dataset_name]['train_ann_path'], 'r')  as f:
        vis_data = json.load(f)
# deformable_detr_single.py
if VISUALIZE:
	Attn_masks_folder = 'Attn_masks_folder/FL_hs/BL_wts_25212_avg_feat_maps_greater-than-0.9_1/'
	os.makedirs(Attn_masks_folder, exist_ok=True)
