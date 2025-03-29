import os

import numpy as np
import torch
from isbnet.util.rle import rle_decode
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from scannetv2_inst_eval import ScanNetEval
from tqdm import tqdm


scan_eval = ScanNetEval(class_labels=INSTANCE_CAT_SCANNET_200)
# data_path = "../exp/version_qualitative/final_result_hier_agglo_2d"
data_path = '../exp_material_1/version_text/final_result_hier_agglo_material_prompt'
# data_path = '../exp/version_text/final_result_hier_agglo'

pcl_path = "./data/Scannet200/Scannet200_3D/val/groundtruth"

## change different attribute gt path
gt_txt_base_path = '/gt_material_openmask3d'

if __name__ == "__main__":
    scenes = sorted([s for s in os.listdir(data_path) if s.endswith(".pth")])

    gtsem = []
    gtinst = []
    res = []

    for scene in tqdm(scenes):

        gt_path = os.path.join(pcl_path, scene)
        loader = torch.load(gt_path)

        gt_txt_path = os.path.join(gt_txt_base_path, scene.replace('pth', 'txt'))
        gt_ids = np.loadtxt(gt_txt_path)

        sem_gt, inst_gt = loader[2], loader[3]

        # print(scene)
        sem_gt= loader[2]
        sem_gt = gt_ids // 1000
        # # gt_ = gt_ids // 1000
        # # sem_gt = [-100 for gt_ in range(1192)]
        inst_gt = gt_ids % 1000
        # print('sem_gt', sem_gt[:100])
        # print('inst_gt_attr',inst_gt[:100])
        # print(loader[3])
        # print(sem_gt)
        # print(loader[2])

        gtsem.append(np.array(sem_gt).astype(np.int32))
        gtinst.append(np.array(inst_gt).astype(np.int32))

        scene_path = os.path.join(data_path, scene)
        pred_mask = torch.load(scene_path)
        # print(scene_path)
        # assert False

        masks, category, score = pred_mask["ins"], pred_mask["final_class"], pred_mask["conf"]

        n_mask = category.shape[0]
        tmp = []
        for ind in range(n_mask):
            if isinstance(masks[ind], dict):
                mask = rle_decode(masks[ind])
            else:
                mask = (masks[ind] == 1).numpy().astype(np.uint8)
            # conf = score[ind] #
            conf = 1.0
            final_class = float(category[ind])
            scene_id = scene.replace(".pth", "")
            tmp.append({"scan_id": scene_id, "label_id": final_class + 1, "conf": conf, "pred_mask": mask})

        res.append(tmp)

    scan_eval.evaluate(res, gtsem, gtinst)
