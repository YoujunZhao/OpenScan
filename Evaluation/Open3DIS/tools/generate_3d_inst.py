import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import clip
import torch
import yaml
from munch import Munch
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200, INSTANCE_CAT_SCANNET_200_1 # Scannet200
from open3dis.dataset.scannetpp import SEMANTIC_CAT_SCANNET_PP # ScannetPP
from open3dis.dataset.replica import INSTANCE_CAT_REPLICA
from open3dis.dataset.s3dis import INSTANCE_CAT_S3DIS, AREA

from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval
# from open3dis.src.clustering.clustering import process_hierarchical_agglomerative
from open3dis.src.clustering.clustering import process_hierarchical_agglomerative_spp, process_hierarchical_agglomerative_nospp

from torch.nn import functional as F
from tqdm import tqdm, trange

import torch.nn as nn
import math
from scipy.stats import kstest

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-12, device = 'cuda'):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        # self.weight = nn.Parameter(torch.ones(hidden_size, device = 'cuda'))
        # self.bias = nn.Parameter(torch.zeros(hidden_size, device = 'cuda'))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return x

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)
        self.LayerNorm = LayerNorm()

    def forward(self, q, k, v, mask=None):
        print(q.size())
        print(k.size())
        q = torch.unsqueeze(q, 0)
        k = torch.unsqueeze(k, 0)
        v = torch.unsqueeze(v, 0)
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output
        # output = 0.5 * torch.bmm(attn, q) + 0.5 * v # 5.Output
        print(output.size())
        # output = self.LayerNorm(output + q)
        # output = output + q
        output = 0.2 * output + 0.8 * q
        output = torch.squeeze(output)

        return attn, output

# class LayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12, device = 'cuda'):
#         """Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(LayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size, device = 'cuda'))
#         self.bias = nn.Parameter(torch.zeros(hidden_size, device = 'cuda'))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size, dtype = torch.half, device = 'cuda')
        self.key = nn.Linear(input_size, self.all_head_size, dtype = torch.half, device = 'cuda')
        self.value = nn.Linear(input_size, self.all_head_size, dtype = torch.half, device = 'cuda')

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size, dtype = torch.half, device = 'cuda')
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12, device = 'cuda')
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        x = x.unsqueeze(0)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor_q, input_tensor_k, input_tensor_v):
        # print('q',input_tensor_q.dtype)
        # print('k',input_tensor_k.dtype)
        # print('v',input_tensor_v.dtype)
        # mixed_query_layer = self.query(input_tensor_q.half())
        # mixed_key_layer = self.key(input_tensor_k.half())
        mixed_query_layer = input_tensor_q.half()
        mixed_key_layer = input_tensor_k.half()
        # mixed_value_layer = self.value(input_tensor_v)
        mixed_value_layer = input_tensor_v.half()

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor_q)
        hidden_states = hidden_states.squeeze(0)
        return hidden_states

def rle_encode_gpu_batch(masks):
    """
    Encode RLE (Run-length-encode) from 1D binary mask.
    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1

        runs[1::2] -= runs[::2]

        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles


def rle_decode(rle):
    """
    Decode rle to get binary mask.
    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except:
        s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def get_final_instances(
    cfg, text_features, cluster_dict=None, use_2d_proposals=False, use_3d_proposals=True, only_instance=True
):
    """
    Get final 3D instance (2D | 3D), point cloud features from which stage
    returning masks, class and scores
    """
    exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)

    pc_features_path = os.path.join(exp_path, cfg.exp.grounded_feat_output, f"{scene_id}.pth")
    pc_refined_features_path = os.path.join(exp_path, cfg.exp.refined_grounded_feat_output, f"{scene_id}.pth")
    
    # Choose which stage to use the feature ?
    if cfg.proposals.refined and os.path.exists(pc_refined_features_path):
        pc_features = torch.load(pc_refined_features_path)["feat"].cuda()
    else:
        pc_features = torch.load(pc_features_path)["feat"].cuda()
    
    pc_features = F.normalize(pc_features, dim=1, p=2)

    # 2D lifting 3D mask path
    cluster_dict_path = os.path.join(exp_path, cfg.exp.clustering_3d_output, f"{scene_id}.pth")

    if cluster_dict is not None:
        data = cluster_dict
    else:
        data = torch.load(cluster_dict_path)

    if isinstance(data["ins"][0], dict):
        instance_2d = torch.stack([torch.from_numpy(rle_decode(ins)) for ins in data["ins"]], dim=0).cuda()
    else:
        instance_2d = data["ins"].cuda()

    confidence_2d = torch.tensor(data["conf"]).cuda()

    ########### Proposal branch selection ###########
    if use_3d_proposals:
        if cfg.data.dataset_name == 's3dis':
            agnostic3d_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path, f"{AREA}_{scene_id}.pth")
        else:
            agnostic3d_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path, f"{scene_id}.pth")
        agnostic3d_data = torch.load(agnostic3d_path)
        instance_3d_encoded = np.array(agnostic3d_data["ins"])
        confidence_3d = torch.tensor(agnostic3d_data["conf"]).cuda()

        n_instance_3d = instance_3d_encoded.shape[0]

        if isinstance(instance_3d_encoded[0], dict):
            instance_3d = torch.stack(
                [torch.from_numpy(rle_decode(in3d)) for in3d in instance_3d_encoded], dim=0
            ).cuda()
        else:
            instance_3d = torch.stack([torch.tensor(in3d) for in3d in instance_3d_encoded], dim=0).cuda()

        ######################## doing nms

        intersection = torch.einsum("nc,mc->nm", instance_2d.float(), instance_3d.float())
        # print(intersection.shape, instance.shape, )
        ious = intersection / (instance_2d.sum(1)[:, None] + instance_3d.sum(1)[None, :] - intersection)

        ious_2d_in_3d = intersection / instance_3d.sum(1)[None, :]
        ious_3d_in_2d = intersection / instance_2d.sum(1)[:, None]

        ious_max, ious_max_indices = torch.max(ious, dim=1)
        # ious_max = torch.max(ious, dim=1)[0]
        ious_max_3d, ious_max_3d_indices = torch.max(ious_3d_in_2d, dim=1)
        ious_max_2d, ious_max_2d_indices = torch.max(ious_2d_in_3d, dim=1)
        print(ious.shape)
        print('ious_max_indices', ious_max_indices)
        print(ious_max.shape)
        print(ious_max_3d.shape)
        print(ious_max_2d.shape)
        print(instance_3d.shape)
        print(instance_2d.shape)
        valid_mask = torch.ones(instance_2d.shape[0], dtype=torch.bool, device=instance_2d.device)

        ious_3d_agg = torch.zeros(instance_3d.shape[0], dtype=torch.half, device=instance_3d.device)
        ious_3d_agg_k = torch.zeros(instance_3d.shape[0], dtype=torch.half, device=instance_3d.device)

        for i in range(len(ious_max)):
            if ious_max[i] >= cfg.final_instance.iou_overlap:
                ious_3d_agg[ious_max_indices[i]] += ious_max[i]
                ious_3d_agg_k[ious_max_indices[i]] += 1    
            if ious_max_3d[i] >= cfg.final_instance.iou_overlap:
                ious_3d_agg[ious_max_3d_indices[i]] += ious_max_3d[i]
                ious_3d_agg_k[ious_max_3d_indices[i]] += 1     
            if ious_max_2d[i] >= cfg.final_instance.iou_overlap:
                ious_3d_agg[ious_max_2d_indices[i]] += ious_max_2d[i]
                ious_3d_agg_k[ious_max_2d_indices[i]] += 1                                

        for j in range(len(ious_3d_agg)):
            if ious_3d_agg_k[j] !=0:
                ious_3d_agg[j] /= ious_3d_agg_k[j]
        print('ious_3d_agg', ious_3d_agg)

        # all IOU > iou_overlap: delete 2D mask 

        valid_mask[ious_max >= cfg.final_instance.iou_overlap] = 0
        print(valid_mask)

        # 2D IOU > iou_overlap: 2D mask in 3D mask: delete 2D mask 

        valid_mask[ious_max_2d >= cfg.final_instance.iou_overlap] = 0

        # 3D IOU > iou_overlap: 3D mask in 2D mask: delete 3D mask 

        # valid_mask_3d = torch.ones(instance_3d.shape[0], dtype=torch.bool, device=instance_3d.device)

        # valid_mask_3d[ious_max_3d >= cfg.final_instance.iou_overlap] = 0
        valid_mask[ious_max_3d >= cfg.final_instance.iou_overlap] = 0


        instance_2d = instance_2d[valid_mask]
        confidence_2d = confidence_2d[valid_mask]
        print(valid_mask.shape)
        print(torch.unique(valid_mask,return_counts=True))
        print('valid_mask', valid_mask)
        print('instance_2d', instance_2d)
        # print(confidence_2d)
        # print(confidence_2d.shape)
        # print(instance_2d.shape)
        # assert False

        # instance_3d = instance_3d[valid_mask_3d]
        # confidence_3d = confidence_3d[valid_mask_3d]
        print('instance_2d', instance_2d.shape)
        print('instance_3d', instance_3d.shape)

        ious_2d_agg = torch.zeros(instance_2d.shape[0], dtype=torch.half, device=instance_2d.device)
        ious_agg = torch.cat((ious_2d_agg, ious_3d_agg))
        print(ious_agg.shape)

        ##########################

    if use_2d_proposals and use_3d_proposals:
        instance = torch.cat([instance_2d, instance_3d], dim=0)
        confidence = torch.cat([confidence_2d, confidence_3d], dim=0)
    elif use_2d_proposals and not use_3d_proposals:
        instance = instance_2d
        confidence = torch.cat([confidence_2d], dim=0)
    else:
        instance = instance_3d
        confidence = torch.cat([confidence_3d], dim=0)
    ########### ########### ########### ###########

    n_instance = instance.shape[0]
    print(instance.shape)
    print('n_instance', n_instance)

    if only_instance == True:  # Return class-agnostic 3D instance
        return instance, None, None

    
    
    ### Offloading CPU for scannetpp @@
    # NOTE Pointwise semantic scores
    predicted_class = (cfg.final_instance.scale_semantic_score * pc_features @ text_features.cuda().T.float()).softmax(dim=-1)

    # predicted_class = torch.zeros((pc_features.shape[0], text_features.shape[0]), dtype = torch.float32)
    # bs = 100000
    # for batch in range(0, pc_features.shape[0], bs):
    #     start = batch
    #     end = min(start + bs, pc_features.shape[0])
    #     predicted_class[start:end] = (cfg.final_instance.scale_semantic_score * pc_features[start:end].cpu() @ text_features.T.cpu().to(torch.float32)).softmax(dim=-1).cpu()

    print(predicted_class.shape)
    print('predicted_class', predicted_class)
    # assert False

    # NOTE Mask-wise semantic scores
    inst_class_scores = torch.einsum("kn,nc->kc", instance.float(), predicted_class.float()).cuda() # K x classes
    print('----')
    print(inst_class_scores)
    print(inst_class_scores.shape)
    print(ious_agg)
    print(ious_agg.shape)
    print(instance.shape)

    inst_class_scores = inst_class_scores / instance.float().cuda().sum(dim=1)[:, None]  # K x classes
    print('inst_class_scores_1', inst_class_scores[0])

    print(torch.sum(inst_class_scores, dim=1))
    beta = 3
    inst_class_scores = (inst_class_scores) * ((ious_agg.unsqueeze(1)+1) ** beta)
    inst_class_scores = inst_class_scores / inst_class_scores.float().cuda().sum(dim=1)[:, None]  # K x classes

    # inst_class_scores = inst_class_scores / instance.float().cuda().sum(dim=1)[:, None]  # K x classes
    print('inst_class_scores_2', inst_class_scores[0])
    print(torch.sum(inst_class_scores, dim=1))

    # assert False
    # # NOTE Top-K instances
    inst_class_scores = inst_class_scores.reshape(-1)  # n_cls * n_queries
    print(inst_class_scores)
    print(inst_class_scores.shape)

    labels = (
        torch.arange(cfg.data.num_classes, device=inst_class_scores.device)
        .unsqueeze(0)
        .repeat(n_instance, 1)
        .flatten(0, 1)
    )


    cur_topk = 800 if use_3d_proposals else cfg.final_instance.top_k
    _, idx = torch.topk(inst_class_scores, k=min(cur_topk, len(inst_class_scores)), largest=True)
    mask_idx = torch.div(idx, cfg.data.num_classes, rounding_mode="floor")
    print(idx)
    print(mask_idx)
    # assert False
    cls_final = labels[idx]
    scores_final = inst_class_scores[idx].cuda()
    masks_final = instance[mask_idx]

    return masks_final, cls_final, scores_final


# evaluate_openvocab = False
# evaluate_agnostic = False

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser

# use 3 sigma to delete abnormal text 
# def KsNormDetect(df):
#     # 计算均值
#     df = df.cpu().numpy()
#     u = df.mean()
#     # 计算标准差
#     std = df.std()
#     # 计算P值
#     print(kstest(df, 'norm', (u, std)))
#     res = kstest(df, 'norm', (u, std))[1]
#     print('均值为：%.2f，标准差为：%.2f' % (u, std))
#     # 判断p值是否服从正态分布，p<=0.05 拒绝原假设 不服从正态分布
#     if res <= 0.05:
#         print('该列数据不服从正态分布')
#         # print("-" * 66)
#         return False
#     else:
#         print('该列数据服从正态分布')
#         return True

# def OutlierDetection(df, ks_res):
#     df = df.cpu().numpy()
#     # 计算均值
#     u = df.mean()
#     # 计算标准差
#     std = df.std()
#     if ks_res:
#         # 定义3σ法则识别异常值
#         print(df)
#         print(df[(df - u) > (-1) * std])
#         clean_data_index = [index for index, value in enumerate(df) if (value - u) > (-1) * std]
#         print(clean_data_index)
#         # outliers = enumerate(df[(df - u) < (-1) * std])
#         # print(outliers)
#         # assert False
#         # 剔除异常值，保留正常的数据
#         # clean_data = enumerate(df[(df - u) > (-1) * std])
#         # 返回异常值和剔除异常值后的数据
#         return clean_data_index

#     else:
#         # print('请先检测数据是否服从正态分布')
#         index, value = enumerate(df)
#         return index
# # if __name__ == '__main__':
# #     # 构造数据  某一列数据  含有异常值
# #     data = np.random.normal(60, 5, 200)
# #     data[6], data[66], data[196] = 16, 360, 180
# #     print(data)
    
# #     print("-" * 66)
# #     # 可以转换为pandas的DataFrame 便于调用方法计算均值和标准差
# #     df = pd.DataFrame(data, columns=['value'])
# #     # box-cox变换
# #     lam = boxcox_normmax(df["value"] + 1)
# #     df["value"] = boxcox1p(df['value'], lam)
# #     # K-S检验
# #     ks_res = KsNormDetect(df)
# #     outliers, clean_data = OutlierDetection(df, ks_res)
# #     # 异常值和剔除异常值后的数据
# #     outliers = inv_boxcox(outliers, lam) - 1
# #     clean_data = inv_boxcox(clean_data, lam) - 1
# #     print(outliers)
# #     print("-" * 66)
# #     print(clean_data)

if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    evaluate_openvocab = cfg.evaluate.evalvocab  # Evaluation for openvocab
    evaluate_agnostic = cfg.evaluate.evalagnostic  # Evaluation for openvocab


    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    # Scannet200 class text features saving
    if cfg.data.dataset_name == 'scannet200':
        class_names = INSTANCE_CAT_SCANNET_200
        class_names_1 = INSTANCE_CAT_SCANNET_200_1
    elif cfg.data.dataset_name == 'scannetpp':
        class_names = SEMANTIC_CAT_SCANNET_PP    
    elif cfg.data.dataset_name == 'replica':
        class_names = INSTANCE_CAT_REPLICA
    elif cfg.data.dataset_name == 's3dis':     
        class_names = INSTANCE_CAT_S3DIS
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

    if evaluate_openvocab:
        scan_eval = ScanNetEval(class_labels=class_names, dataset_name=cfg.data.dataset_name)
        gtsem = []
        gtinst = []
        res = []
        
    clip_adapter, clip_preprocess = clip.load(cfg.foundation_model.clip_model, device = 'cuda')
    # print(class_names)
    # assert False

    # attention_llm = SelfAttention(num_attention_heads = 8, input_size = 768, hidden_size = 768, hidden_dropout_prob = 0.2)
    # attention_llm = ScaledDotProductAttention(scale=np.power(768, 0.5))
    # attention_llm_1 = ScaledDotProductAttention(scale=np.power(768, 0.5))


    with torch.no_grad(), torch.cuda.amp.autocast():

        text_features_0 = clip_adapter.encode_text(clip.tokenize(class_names).cuda())
        # print(text_features_0.size)
        text_features_0 /= text_features_0.norm(dim=-1, keepdim=True)
        # print(class_names)
        # print(class_names_1)
        # print(cfg.data.dataset_name)

        class_names_2 = class_names_1

        # adaptive weight

        split_sim_avg = []

        text_feats = text_features_0

        # for i in range(len(class_names)):
        #     split_class_name_1 = class_names_1[i].split(', ')
        #     print(split_class_name_1)
        #     split_class_name_2 = split_class_name_1
        #     for j in range(len(split_class_name_1)):

        #         split_class_name_2[j] = class_names[i] + split_class_name_1[j]
        #     # class_names_2[i] = split_class_name_2
        #     class_names_2[i] = ', '.join(split_class_name_2)

            # # using clip text encoder to calculate text similarity
            # text_features_split_class_name_1 = clip_adapter.encode_text(clip.tokenize(split_class_name_1).cuda())
            # text_features_split_class_name_1 /= text_features_split_class_name_1.norm(dim=-1, keepdim=True)

            # split_sim_0 = text_features_0[i] @ text_features_split_class_name_1.cuda().T.float()

            #---------------------------------------------------------------------------------
            # using bert to calculate text similarity

            # from transformers import BertTokenizer, BertModel
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # model = BertModel.from_pretrained('bert-base-uncased')

            # def encode_text(text):
            #     inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            #     outputs = model(**inputs)
            #     return outputs.last_hidden_state.mean(dim=1)

            # from sklearn.metrics.pairwise import cosine_similarity

            # main_text = class_names[i]
            # other_texts = split_class_name_1

            # # main_text = 'store clothes'
            # # other_texts = ['closet', 'dresser', 'drawer', 'stove', 'wardrobe']

            # main_embedding = encode_text(main_text)
            # other_embeddings = torch.cat([encode_text(text) for text in other_texts])

            # similarities = cosine_similarity(main_embedding.detach().numpy(), other_embeddings.detach().numpy())
            # split_sim_0 = torch.Tensor(similarities).cuda().squeeze(0)

            #---------------------------------------------------------------------------------
        #     # print(f"bert simi: {similarities}")
        #     # print(np.where(similarities = min(similarities[0])))
        #     # print(f"clip simi: {split_sim_0.shape}")
        #     # from transformers import BertTokenizer, BertModel
        #     # import torch
        #     # import torch.nn.functional as F

        #     # # 定义要比较的单词
        #     # Text_1 = class_names[i]
        #     # Text_2 = split_class_name_1[0]
        #     # print(Text_1)
        #     # print(Text_2)

        #     # # 初始化 BERT tokenizer 和模型
        #     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #     # model = BertModel.from_pretrained('bert-base-uncased')

        #     # # 对单词进行编码
        #     # tokens_a = tokenizer(Text_1, return_tensors='pt')
        #     # tokens_b = tokenizer(Text_2, return_tensors='pt')

        #     # # 获取单词的嵌入
        #     # embeddings= []
        #     # with torch.no_grad():
        #     #     embeddings_a = model(**tokens_a).last_hidden_state.mean(dim=1)
        #     #     embeddings_b = torch.cat(embeddings.append(model(**tokens_b).last_hidden_state.mean(dim=1)))

        #     # # 计算余弦相似度
        #     # cosine_similarity = F.cosine_similarity(embeddings_a, embeddings_b)
        #     # cosine_similarity.item()
        #     # print(cosine_similarity)
        #     # print("clip similarity:", split_sim_0)

        #     # from sklearn.feature_extraction.text import CountVectorizer
        #     # from sklearn.metrics.pairwise import cosine_similarity

        #     # def calculate_cosine_similarity(text1, text2):
        #     #     vectorizer = CountVectorizer()
        #     #     corpus = [text1, text2]
        #     #     vectors = vectorizer.fit_transform(corpus)
        #     #     similarity = cosine_similarity(vectors)
        #     #     return similarity[0][1]

        #     # text1 = class_names[i]
        #     # text2 = split_class_name_1[0]
        #     # # text1 = "I love Python programming"
        #     # # text2 = "Python programming is great"
        #     # cosine_similarity = calculate_cosine_similarity(text1, text2)
        #     # # print(cosine_similarity)
        #     # print(text1)
        #     # print(text2)

        #     # from sentence_transformers import SentenceTransformer
        #     # def calculate_bert_similarity(text1, text2):
        #     #     model = SentenceTransformer("bert-base-nli-mean-tokens")
        #     #     embeddings = model.encode([text1, text2])
        #     #     similarity = cosine_similarity(embeddings)
        #     #     return similarity[0][1]

        #     # # text1 = "I love Python programming"
        #     # # text2 = "Python programming is great"

        #     # bert_similarity = calculate_bert_similarity(text1, text2)
        #     # print(bert_similarity)

        #     # import gensim.downloader as api
        #     # from gensim import matutils
        #     # import numpy as np

        #     # # def calculate_word2vec_similarity(text1, text2):
        #     # #     model = api.load("word2vec-google-news-300")
        #     # #     tokens1 = text1.split()
        #     # #     tokens2 = text2.split()
        #     # #     vec1 = np.mean([model[token] for token in tokens1 if token in model], axis=0)
        #     # #     vec2 = np.mean([model[token] for token in tokens2 if token in model], axis=0)
        #     # #     return matutils.cossim(vec1, vec2)

        #     # # text1 = "I love Python programming"
        #     # # text2 = "Python programming is great"

        #     # # word2vec_similarity = calculate_word2vec_similarity(text1, text2)
        #     # # print(word2vec_similarity)

        #     # assert False

        #     # print(split_sim_0)
        #     # ratio = 1 / sum(split_sim_0)
        #     # split_sim = ratio * split_sim_0
        #     # print(split_sim)

        #     # filter sim > 0.65
        #     df = split_sim_0
        #     clean_data = [index for index, value in enumerate(df) if value > 0.63]
        #     # split_class_name_after = []
        #     # split_sim_after = []
        #     # for j in clean_data:
        #     #     split_class_name_after.append(split_class_name_1[j])
        #     #     split_sim_after.append(split_sim_0[j])
        #     # split_sim_after = torch.Tensor(split_sim_after).cuda()
        #     # # print(split_sim_after)
        #     # # ratio = 1 / sum(split_sim_after)
        #     # # split_sim_avg.append(sum(split_sim_after) / split_sim_after.size(0))
        #     # # print('split_sim_avg', split_sim_avg)
        #     # # assert False

        #     # text_features_split_class_name_1 = clip_adapter.encode_text(clip.tokenize(split_class_name_after).cuda())
        #     # text_features_split_class_name_1 /= text_features_split_class_name_1.norm(dim=-1, keepdim=True)
        #     # print(text_features_split_class_name_1.shape)
        #     # assert False
        #     # print(len(split_class_name_1))
        #     # print(split_class_name_1)
        #     # split_sim_0 = text_features_0[i] @ text_features_split_class_name_1.cuda().T.float()
        #     # print(split_sim_0)
        #     # ratio = 1 / sum(split_sim_0)
        #     # split_sim = ratio * split_sim_0
        #     # print(split_sim)
        #     # assert False
        #     # print(text_feats.shape)
        #     # text_feats[i, :] = split_sim_avg @ text_features_split_class_name_1

        # # filter sim > 0.65

        #     # df = split_sim.cpu()
        #     # print(df)
        #     # clean_data = [index for index, value in enumerate(df) if value > 0.63]

        #     split_class_name_after = []

        #     # adaptive weight
        #     split_sim_after = []

        #     for j in clean_data:
        #         split_class_name_after.append(split_class_name_1[j])

        #         # adaptive weight
        #         split_sim_after.append(df[j])
        #     print('split_sim_after', split_sim_after)

        #     class_names_2[i] = ', '.join(split_class_name_after)

        #     # adaptive weight
        #     # split_sim_avg.append(np.mean(split_sim_after))
            
        # print('class_names_2', class_names_2)

        #---------------------------------------------------------------------------------
        text_features_1 = clip_adapter.encode_text(clip.tokenize(class_names_2).cuda())
        text_features_1 /= text_features_1.norm(dim=-1, keepdim=True)

        # print(split_sim_avg)
        # assert False
        # # attention features fusion
        # # text_features = attention_llm(text_features_0, text_features_1, text_features_1)
        # # mask = torch.zeros(batch, n_q, n_k).bool()
        # # q, k, v = text_features_1, text_features_1, text_features_0
        # # mask = torch.zeros(1, q.size(0), k.size(0), device = 'cuda').bool()
        # attn, output = attention_llm(text_features_1, text_features_1, text_features_1, mask=None)
        # attn_1, output_1 = attention_llm_1(text_features_0, text_features_0, text_features_0, mask=None)

        # text_features = 0.5 * output + 0.5 * output_1
        # # print(output)
        # # assert False

        # adaptive weight
        split_sim_avg = torch.Tensor(split_sim_avg)
        # print(split_sim_avg.size(1))

        split_sim_avg = split_sim_avg.expand(split_sim_avg.size(0), split_sim_avg.size(0)).cuda()
        # print('split_sim_avg', split_sim_avg)
        # assert False  

        # Add features fusion
        # text_features = 0.5 * text_features_0 + 0.5 * text_features_1
        text_features = 0.5 * text_features_0 + 0.2 * text_features_1
        # text_features = 0.5 * text_features_0 + 0.3 * split_sim_avg @ text_features_1
        # text_features = text_features_0


        # adaptive weight
        # text_features = 0.5 * text_features_0 + split_sim_avg * text_features_1

        # text_features = text_features_0 + 0.65 * text_features_1
        # text_features = text_features_1

        # Weight features fusion
        # text_features = text_features_0 + text_feats

    # Prepare directories
    save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    os.makedirs(save_dir_cluster, exist_ok=True)
    save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output) # final_output
    os.makedirs(save_dir_final, exist_ok=True)

    # Multiprocess logger
    if os.path.exists("tracker_lifted.txt") == False:
        with open("tracker_lifted.txt", "w") as file:
            file.write("Processed Scenes .\n")

    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            print("Process", scene_id)
            # # Tracker

            # done = False
            # path = scene_id + ".pth"
            # with open("tracker_lifted.txt", "r") as file:
            #     lines = file.readlines()
            #     lines = [line.strip() for line in lines]
            #     for line in lines:
            #         if path in line:
            #             done = True
            #             break
            # if done == True:
            #     print("existed " + path)
            #     continue
            # ## Write append each line
            # with open("tracker_lifted.txt", "a") as file:
            #     file.write(path + "\n")

            # if os.path.exists(os.path.join(save_dir_final, f"{scene_id}.pth")): 
            #     print(f"Skip {scene_id} as it already exists")
            #     continue

            #############################################
            # NOTE hierarchical agglomerative clustering
            if False:
                cluster_dict = None
                proposals3d, confidence = process_hierarchical_agglomerative(scene_id, cfg)

                if proposals3d == None: # Discarding too large scene
                    continue

                cluster_dict = {
                    "ins": rle_encode_gpu_batch(proposals3d),
                    "conf": confidence,
                }
                torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))

            #############################################
            # NOTE get final instances
            # if True:   
            #     cluster_dict = torch.load(os.path.join(save_dir_cluster, f"{scene_id}.pth"))
            #     masks_final, cls_final, scores_final = get_final_instances(
            #         cfg,
            #         text_features,
            #         cluster_dict=cluster_dict,
            #         use_2d_proposals=cfg.proposals.p2d,
            #         use_3d_proposals=cfg.proposals.p3d,
            #         only_instance=cfg.proposals.agnostic,
            #     )
            #     # print(cls_final)
            #     # assert False
            #     final_dict = {
            #         "ins": rle_encode_gpu_batch(masks_final),
            #         "conf": scores_final.cpu(),
            #         "final_class": cls_final.cpu(),
            #     }
            #     # NOTE Final instance
            #     torch.save(final_dict, os.path.join(save_dir_final, f"{scene_id}.pth"))

            if True:   
                cluster_dict = torch.load(os.path.join(save_dir_cluster, f"{scene_id}.pth"))
                masks_final, cls_final, scores_final = get_final_instances(
                    cfg,
                    text_features,
                    cluster_dict=cluster_dict,
                    use_2d_proposals=cfg.proposals.p2d,
                    use_3d_proposals=cfg.proposals.p3d,
                    only_instance=cfg.proposals.agnostic,
                )
                if scores_final == None:
                    final_dict = {
                        "ins": rle_encode_gpu_batch(masks_final),
                        "conf": None,
                        "final_class": None,
                    }
                else:
                    final_dict = {
                        "ins": rle_encode_gpu_batch(masks_final),
                        "conf": scores_final.cpu(),
                        "final_class": cls_final.cpu(),
                    }
                # NOTE Final instance
                torch.save(final_dict, os.path.join(save_dir_final, f"{scene_id}.pth"))
            #############################################
            # NOTE Evaluation openvocab
            if evaluate_openvocab:
                
                if cfg.data.dataset_name == 's3dis':
                    gt_path = os.path.join(cfg.data.gt_pth, f"{AREA}_{scene_id}.pth")
                    _, _, sem_gt, inst_gt = torch.load(gt_path)
                    n_points = len(sem_gt)
                    if n_points > 1000000:
                        stride = 8
                    elif n_points >= 600000:
                        stride = 6
                    elif n_points >= 400000:
                        stride = 2
                    else:
                        stride = 1
                    sem_gt = sem_gt[::stride]
                    inst_gt = inst_gt[::stride]

                    # NOTE do not eval class "clutter"
                    inst_gt[sem_gt==12] = -100
                    sem_gt[sem_gt==12] = -100
                else:
                    gt_path = os.path.join(cfg.data.gt_pth, f"{scene_id}.pth")
                    _, _, sem_gt, inst_gt = torch.load(gt_path)

                gtsem.append(np.array(sem_gt).astype(np.int32))
                gtinst.append(np.array(inst_gt).astype(np.int32))

                masks_final = masks_final.cpu()
                cls_final = cls_final.cpu()

                n_mask = masks_final.shape[0]
                tmp = []
                for ind in range(n_mask):
                    mask = (masks_final[ind] == 1).numpy().astype(np.uint8)
                    conf = 1.0  # Same as OpenMask3D
                    final_class = float(cls_final[ind])
                    tmp.append({"scan_id": scene_id, "label_id": final_class + 1, "conf": conf, "pred_mask": mask})
                res.append(tmp)
            # NOTE Evaluation agnostic
            if evaluate_agnostic:
                pass

            print("Done")
            torch.cuda.empty_cache()

        if evaluate_openvocab:
            scan_eval.evaluate(
                res, gtsem, gtinst, exp_path=os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output)
            )
        if evaluate_agnostic:
            pass
