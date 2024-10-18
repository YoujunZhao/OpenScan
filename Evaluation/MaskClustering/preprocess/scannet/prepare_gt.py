from tqdm import tqdm
import os
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from evaluation.constants import SCANNET_IDS

raw_data_dir = '../../ScanNet/scans'
#        ###########
gt_dir = '../../data/scannet/gt_type_openmask3d'
#        ###########

label_map_file = '../../ScanNet/scannetv2-labels.combined.tsv'

#        ###########
label_map_material_file = '../../ScanNet/scannetv2-labels.type.tsv'
#        ###########
label_material_json = '../../ScanNet/openscan_7.11_val_type.json'
#        ###########
split_file_path = '../../splits/scannet.txt'

CLOUD_FILE_PFIX = '_vh_clean_2'
SEGMENTS_FILE_PFIX = '.0.010000.segs.json'
AGGREGATIONS_FILE_PFIX = '.aggregation.json'

def export_gt(filename, label_ids, instance_ids):
    # print('label_ids', label_ids)
    # print('instance_ids', max(instance_ids))
    gt_data = label_ids * 1000 + instance_ids + 1
    # assert False
    np.savetxt(filename, gt_data, fmt='%d')

# Map the raw category id to the point cloud
def point_indices_from_group(seg_indices, group, labels_pd):
    group_segments = np.array(group['segments'])
    label = group['label']
    # print('label:', label)
    # print('labels_pd', labels_pd)
    # assert False

    # Map the category name to id
    label_ids = labels_pd[labels_pd['raw_category'] == label]['id']

    # print('label_ids:', label_ids)
    # assert False
    label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0
    # print('label_ids:', label_ids)
    # assert False
    # Only store for the valid categories
    if not label_id in SCANNET_IDS:
        label_id = 0

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_IDs = np.where(np.isin(seg_indices, group_segments))
    # print('point_IDs:', point_IDs)
    # assert False
    return point_IDs[0], label_id

def handle_process(scene_path, output_path, labels_pd, labels_material):
    scene_id = scene_path.split('/')[-1]
    print('scene id:', scene_id)
    segments_file = os.path.join(scene_path, f'{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}')
    aggregations_file = os.path.join(scene_path, f'{scene_id}{AGGREGATIONS_FILE_PFIX}')

    output_gt_file = os.path.join(output_path, f'{scene_id}.txt')
 
    # Load segments file
    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments['segIndices'])

    # Load Aggregations file
    with open(aggregations_file) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation['segGroups'])

    # Load Material file
    with open(label_material_json) as f:
        aggregation_material = json.load(f)  

    # Generate new labels
    labelled_pc = np.zeros((len(seg_indices), 1))
    instance_ids = np.zeros((len(seg_indices), 1))
    for group in seg_groups:
        p_inds, label_id = point_indices_from_group(seg_indices, group, labels_pd)

        # labelled_pc[p_inds] = label_id

        ########### openmask3d:        
        instance_ids[p_inds] = group['id']
        ########### maskclustering:
        # instance_ids[p_inds] = group['id'] + 1
        ###########
        # print('group id', group['id'])
        # print(len(aggregation_material))
        for i in aggregation_material:
            # print(type(i['scene_id']))
            # print(type(i['object_id']))
            # print(type(scene_id))
            # print(type(group['id']))
            # assert False
            if i['scene_id'] == scene_id and int(i['object_id']) == group['id']:
                # material_label = i['material'][0]
                ###########
                material_label = i['type']
                ###########
                # print(labels_material['material_category'].replace('_', ' '))
                # assert False
                material_id = labels_material[labels_material['material_category'] == material_label.replace(' ', '_')]['id']
                print('material_label', material_label)
                print('material_id', material_id)
                # assert False
        # print('material', aggregation_material['scene_id' == scene_id]['object_id' == group['id']]['material'])
        # assert False
        # print(material_id)
        labelled_pc[p_inds] = material_id

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)

    export_gt(output_gt_file, labelled_pc, instance_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=1, type=int, help='The number of parallel workers')
    config = parser.parse_args()

    # Load label map
    labels_pd = pd.read_csv(label_map_file, sep='\t', header=0)
    labels_material = pd.read_csv(label_map_material_file, sep=' ', header=0)
    # print(labels_pd)
    # print(labels_material)

    with open(split_file_path) as val_file:
        val_scenes = val_file.read().splitlines()

    # Load scene paths
    scene_paths = [os.path.join(raw_data_dir, scene) for scene in val_scenes]

    os.makedirs(gt_dir, exist_ok=True)

    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    print('Processing scenes...')
    # show progress
    _ = list(tqdm(pool.map(handle_process, scene_paths, repeat(gt_dir), repeat(labels_pd), repeat(labels_material)), total=len(scene_paths)))