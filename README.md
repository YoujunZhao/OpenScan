
<p align="center">

  <h1 align="center">OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding</h1>
  <p align="center">
    <a href="https://youjunzhao.github.io/">Youjun Zhao</a><sup>1</sup>,
    <a href="https://jiaying.link/">Jiaying Lin</a><sup>1</sup>,
    <a href="https://shuquanye.com/">Shuquan Ye</a><sup>1</sup>, 
    <a href="https://qspang.github.io/">Qianshi Pang</a><sup>2</sup>, 
    <a href="https://www.cs.cityu.edu.hk/~rynson/">Rynson W.H. Lau</a><sup>1</sup>
    <br>
    <sup>1</sup>City University of Hong Kong, 
    <sup>2</sup>South China University of Technology
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2408.11030">📄[arXiv]</a> | <a href="https://youjunzhao.github.io/OpenScan/">🔥[Project]</a> | <a href="https://github.com/YoujunZhao/OpenScan">💻[Code]</a> | <a href="https://github.com/YoujunZhao/OpenScan?tab=readme-ov-file#benchmark-installation">🧩[Data]</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
<strong>OpenScan</strong> is a novel benchmark that facilitates comprehensive evaluation of the generalization ability of 3D scene understanding models on abstract object attributes. It expands the single category of object classes in ScanNet200 into eight linguistic aspects of object-related attributes.
</p>
<br>

<div align="center">
  <img src="https://github.com/YoujunZhao/OpenScan/blob/main/imgs/dataset_vis.jpg?raw=true" width="100%" height="100%"/>
</div><br/>

## News

* **18 Oct 2024**: Release the evaluationn code of OpenScan benchmark. 💻
* **27 Aug 2024**: Release the validation set of OpenScan benchmark. 🧩
* **20 Aug 2024**: [OpenScan](https://arxiv.org/abs/2408.11030) released on arXiv. 📝

## Benchmark Installation
If you want to download the OpenScan benchmark data, we provide the raw validation set from [OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/youjzhao2-c_my_cityu_edu_hk/ETXoQ8QIZNpKnxCLvtT8Xl8BWAcvo_SoiAHd_ao3is1cKQ?e=mWHwbS), and the label mapping file from [OneDrive](https://portland-my.sharepoint.com/:f:/g/personal/youjzhao2-c_my_cityu_edu_hk/EtTl_Lv8-idGtgjn-b1jpKUBulI_pQ1TonnV3Ypu8WUqYw?e=iIucfW).

You can also download the processed validation set from [OneDrive](https://portland-my.sharepoint.com/:f:/g/personal/youjzhao2-c_my_cityu_edu_hk/EiIUDBGAK7FNrtT_zpYNAaQBEv8HTjZPNwJDQYBGl6g6YQ?e=5P1n9M).

## Benchmark Format

```python
    {
        "scene_id":   [ScanNet scene id,           e.g. "scene0011_00"],
        "object_id":  [ScanNet object id,          e.g. "0"],
        "object_name":[ScanNet object name,        e.g. "chair"],
        "material":   [ScanNet object material,    e.g. "wood"],
        "affordance": [ScanNet object affordance,  e.g. "sleep"],
        "property":   [ScanNet object property,    e.g. "soft"],
        "type":       [ScanNet object type,        e.g. "source of illumination"],
        "manner":     [ScanNet object manner,      e.g. "steered by handlebars"],
        "synonyms":   [ScanNet object synonyms,    e.g. "bedside table"],
        "requirement":[ScanNet object requirement, e.g. "water and sun"],
        "element":    [ScanNet object element,     e.g. "88 keys"]
    },

```

## Evaluation

### 1. Quick Evaluation on Your Codebase
If your codebase already supports evaluation for the ScanNet or ScanNet200 benchmarks, you can easily adapt it for the OpenScan benchmark by changing the ground truth (GT) labels and label mapping files.

* Download the processed validation set and the label mapping file for the OpenScan benchmark from [Benchmark Installation](https://github.com/YoujunZhao/OpenScan?tab=readme-ov-file#benchmark-installation).

* Place the processed OpenScan validation set into your GT file directory.

* Replace your existing label mapping scripts with the OpenScan label mapping file (e.g, replace the [SCANNET_LABELS and SCANNET_IDS](https://github.com/YoujunZhao/OpenScan/blob/main/Evaluation/MaskClustering/evaluation/constants_material_1.py)).

* Run your evaluation process.

### 2. Evaluation on Existing 3D Scene Understanding Baselines

* For OpenMask3D, please refer to [OpenMask3D](https://github.com/YoujunZhao/OpenScan/tree/main/Evaluation/openmask3d_sai3d).

* For SAI3D, please refer to [SAI3D](https://github.com/YoujunZhao/OpenScan/tree/main/Evaluation/openmask3d_sai3d).

* For MaskClustering, please refer to [MaskClustering](https://github.com/YoujunZhao/OpenScan/tree/main/Evaluation/MaskClustering).

* For Open3DIS, please refer to [Open3DIS](https://github.com/YoujunZhao/OpenScan/tree/main/Evaluation/Open3DIS).


## Citation :pray:
```
@article{zhao2024openscan,
  title={OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding},
  author={Zhao, Youjun and Lin, Jiaying and Ye, Shuquan and Pang, Qianshi and Lau, Rynson WH},
  journal={arXiv preprint arXiv:2408.11030},
  year={2024}
}
```
