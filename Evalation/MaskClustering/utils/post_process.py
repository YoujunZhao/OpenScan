import numpy as np
import os
import torch
from utils.geometry import judge_bbox_overlay


def merge_overlapping_objects(total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio):
    '''
        Merge objects that have larger than 0.8 overlapping ratio.
    '''
    total_object_num = len(total_point_ids_list)
    invalid_object = np.zeros(total_object_num, dtype=bool)

    for i in range(total_object_num):
        if invalid_object[i]:
            continue
        point_ids_i = set(total_point_ids_list[i])
        bbox_i = total_bbox_list[i]
        for j in range(i+1, total_object_num):
            if invalid_object[j]:
                continue
            point_ids_j = set(total_point_ids_list[j])
            bbox_j = total_bbox_list[j]
            if judge_bbox_overlay(bbox_i, bbox_j):
                intersect = len(point_ids_i.intersection(point_ids_j))
                if intersect / len(point_ids_i) > overlapping_ratio:
                    invalid_object[i] = True
                elif intersect / len(point_ids_j) > overlapping_ratio:
                    invalid_object[j] = True

    valid_point_ids_list = []
    valid_pcld_mask_list = []
    for i in range(total_object_num):
        if not invalid_object[i]:
            valid_point_ids_list.append(total_point_ids_list[i])
            valid_pcld_mask_list.append(total_mask_list[i])
    return valid_point_ids_list, valid_pcld_mask_list


def filter_point(point_frame_matrix, node, pcld_list, point_ids_list, mask_point_clouds, frame_list, args):
    '''
        Following OVIR-3D, we filter the points that hardly appear in this cluster (node), i.e. the detection ratio is lower than a threshold.
        Specifically, detection ratio = #frames that the point appears in this cluster (node) / #frames that the point appears in the whole video.
    '''
    def count_point_appears_in_video(point_frame_matrix, point_ids_list, node_global_frame_id_list):
        '''
            For all points in the cluster, compute #frames that the point appears in the whole video.
            Initialize #frames that the point appears in this cluster as 0.
        '''
        point_appear_in_video_nums, point_appear_in_node_matrixs = [], []
        for point_ids in point_ids_list:
            point_appear_in_video_matrix = point_frame_matrix[point_ids, ]
            point_appear_in_video_matrix = point_appear_in_video_matrix[:, node_global_frame_id_list]
            point_appear_in_video_nums.append(np.sum(point_appear_in_video_matrix, axis=1))
            
            point_appear_in_node_matrix = np.zeros_like(point_appear_in_video_matrix, dtype=bool) # initialize as False
            point_appear_in_node_matrixs.append(point_appear_in_node_matrix)
        return point_appear_in_video_nums, point_appear_in_node_matrixs

    def count_point_appears_in_node(mask_list, node_frame_id_list, point_ids_list, mask_point_clouds, point_appear_in_node_matrixs):
        '''
            Fillin the point_appear_in_node_matrixs by iterating the masks in this cluster (node).
            Meanwhile, since we split the disconnected point cloud into different objects, we also decide which object this mask belongs to.
            Besides, for each mask, we compute the coverage of this mask of the object it belongs to for furture use in OpenMask3D.
        '''
        object_mask_list = [[] for _ in range(len(point_ids_list))]

        for frame_id, mask_id in mask_list:
            frame_id_in_list = np.where(node_frame_id_list == frame_id)[0][0]
            mask_point_ids = list(mask_point_clouds[f'{frame_id}_{mask_id}'])

            object_id_with_largest_intersect, largest_intersect, coverage = -1, 0, 0
            for i, point_ids in enumerate(point_ids_list):
                point_ids_within_object = np.where(np.isin(point_ids, mask_point_ids))[0]
                point_appear_in_node_matrixs[i][point_ids_within_object, frame_id_in_list] = True
                if len(point_ids_within_object) > largest_intersect:
                    object_id_with_largest_intersect, largest_intersect = i, len(point_ids_within_object)
                    coverage = len(point_ids_within_object) / len(point_ids)
            if largest_intersect == 0:
                continue
            object_mask_list[object_id_with_largest_intersect] += [(frame_id, mask_id, coverage)]
        return object_mask_list, point_appear_in_node_matrixs

    node_global_frame_id_list = torch.where(node.visible_frame)[0].cpu().numpy()
    node_frame_id_list = np.array(frame_list)[node_global_frame_id_list]
    mask_list = node.mask_list
    
    point_appear_in_video_nums, point_appear_in_node_matrixs = count_point_appears_in_video(point_frame_matrix, point_ids_list, node_global_frame_id_list)
    object_mask_list, point_appear_in_node_matrixs = count_point_appears_in_node(mask_list, node_frame_id_list, point_ids_list, mask_point_clouds, point_appear_in_node_matrixs)

    # filter points
    filtered_point_ids, filtered_mask_list, filtered_bbox_list = [], [], []
    for i, (point_appear_in_video_num, point_appear_in_node_matrix) in enumerate(zip(point_appear_in_video_nums, point_appear_in_node_matrixs)):
        detection_ratio = np.sum(point_appear_in_node_matrix, axis=1) / (point_appear_in_video_num + 1e-6)
        valid_point_ids = np.where(detection_ratio > args.point_filter_threshold)[0]
        if len(valid_point_ids) == 0 or len(object_mask_list[i]) < 2:
            continue
        filtered_point_ids.append(point_ids_list[i][valid_point_ids])
        filtered_bbox_list.append([np.amin(pcld_list[i].points, axis=0), np.amax(pcld_list[i].points, axis=0)])
        filtered_mask_list.append(object_mask_list[i])
    return filtered_point_ids, filtered_bbox_list, filtered_mask_list


def dbscan_process(pcld, point_ids, DBSCAN_THRESHOLD=0.1):
    '''
        Following OVIR-3D, we use DBSCAN to split the disconnected point cloud into different objects.
    '''
    
    labels = np.array(pcld.cluster_dbscan(eps=DBSCAN_THRESHOLD, min_points=4)) + 1 # -1 for noise
    count = np.bincount(labels)

    # split disconnected point cloud into different objects
    pcld_list, point_ids_list = [], []
    pcld_ids_list = np.array(point_ids)
    for i in range(len(count)):
        remain_index = np.where(labels == i)[0]
        if len(remain_index) == 0:
            continue
        new_pcld = pcld.select_by_index(remain_index)
        point_ids = pcld_ids_list[remain_index]
        pcld_list.append(new_pcld)
        point_ids_list.append(point_ids)
    return pcld_list, point_ids_list


def find_represent_mask(mask_info_list):
    mask_info_list.sort(key=lambda x: x[2], reverse=True)
    return mask_info_list[:5]


def export_class_agnostic_mask(args, class_agnostic_mask_list):
    pred_dir = os.path.join('data/prediction', args.config)
    os.makedirs(pred_dir, exist_ok=True)

    num_instance = len(class_agnostic_mask_list)
    pred_masks = np.stack(class_agnostic_mask_list, axis=1)
    pred_dict = {
        "pred_masks": pred_masks, 
        "pred_score":  np.ones(num_instance),
        "pred_classes" : np.zeros(num_instance, dtype=np.int32)
    }
    class_agnostic_pred_dir = os.path.join('data/prediction', args.config + '_class_agnostic')
    os.makedirs(class_agnostic_pred_dir, exist_ok=True)
    np.savez(os.path.join(class_agnostic_pred_dir, f'{args.seq_name}.npz'), **pred_dict)
    return


def export(dataset, total_point_ids_list, total_mask_list, args):
    '''
        Export class agnostic masks in standard evaluation format 
        and object dict with corresponding mask lists for semantic instance segmentation.
        Node that after clustering, a node = a cluster of masks = an object.
    '''
    total_point_num = dataset.get_scene_points().shape[0]
    class_agnostic_mask_list = []
    object_dict = {}
    for i, (point_ids, mask_list) in enumerate(zip(total_point_ids_list, total_mask_list)):
        object_dict[i] = {
            'point_ids': point_ids,
            'mask_list': mask_list,
            'repre_mask_list': find_represent_mask(mask_list),
        }
        binary_mask = np.zeros(total_point_num, dtype=bool)
        binary_mask[list(point_ids)] = True
        class_agnostic_mask_list.append(binary_mask)

    export_class_agnostic_mask(args, class_agnostic_mask_list)

    os.makedirs(os.path.join(dataset.object_dict_dir, args.config), exist_ok=True)
    np.save(os.path.join(dataset.object_dict_dir, args.config, 'object_dict.npy'), object_dict, allow_pickle=True)


def post_process(dataset, node_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, args):
    if args.debug:
        print('start exporting')
    
    
    # For each cluster, we follow OVIR-3D to i) use DBScan to split the disconnected point cloud into different objects
    # ii) filter the points that hardly appear within this cluster, i.e. the detection ratio is lower than a threshold
    total_point_ids_list, total_bbox_list, total_mask_list = [], [], []
    for node in (node_list):
        if len(node.mask_list) < 2: # objects merged from less than 2 masks are ignored
            continue
        
        pcld, point_ids = node.get_point_cloud(scene_points)
        pcld_list, point_ids_list = dbscan_process(pcld, point_ids) # split the disconnected point cloud into different objects
        point_ids_list, bbox_list, mask_list = filter_point(point_frame_matrix, node, pcld_list, point_ids_list, mask_point_clouds, frame_list, args)

        total_point_ids_list.extend(point_ids_list)
        total_bbox_list.extend(bbox_list)
        total_mask_list.extend(mask_list)

    # merge objects that have larger than 0.8 overlapping ratio
    total_point_ids_list, total_mask_list = merge_overlapping_objects(total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio=0.8)
    export(dataset, total_point_ids_list, total_mask_list, args)
    return