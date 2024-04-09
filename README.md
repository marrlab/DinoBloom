
| <img src="media/logo.png" width="250" title="Cutie" /> | <h1>DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology</h1> 
|-|-|



Repository of [DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology](https://arxiv.org/abs/2404.05022) that uses [DINOv2](https://arxiv.org/abs/2304.07193) and is adapted from their [original Github repository](https://github.com/facebookresearch/dinov2/tree/main/dinov2). DinoBloom is a model family (ViTs) trained on a large cohort of 13 diverse publicly available datasets of single cells in peripheral blood and bone marrow. The trained models in the can be downloaded on [zenodo](https://zenodo.org/records/10908163) in the variants DinoBloom-S, DinoBloom-B, DinoBloom-L and DinoBloom-G. We show that our models outperforms existing medical and non-medical vision models in (i) linear probing and k-nearest neighbor evaluations for cell-type classification on peripheral blood and bone marrow smears and (ii) weakly supervised multiple instance learning for acute myeloid leukemia subtyping by a large margin.
## Data and pipeline overview
<img src="media/overview.png" width="2000" title="Overview" /> 

## Model farm
| Model         | Feature dim | #params | Weights
|---------------|-------------|---------|---------|
| DinoBloom-S   | 384         | 22M     |[Download](https://zenodo.org/records/10908163/files/DinoBloom-S.pth?download=1)|
| DinoBloom-B   | 768         | 86M     |[Download](https://zenodo.org/records/10908163/files/DinoBloom-B.pth?download=1)|
| DinoBloom-L   | 1024        | 304M    |[Download](https://zenodo.org/records/10908163/files/DinoBloom-L.pth?download=1)|
| DinoBloom-G   | 1536        | 1136M   |[Download](https://zenodo.org/records/10908163/files/DinoBloom-G.pth?download=1)|

To train the model you need to specify the folder with .txt files holding the paths of the images you want to use to train in dinov2/configs/train/custom.yaml
for training on a single GPU run: 
```
python dinov2/train/train.py --config-file dinov2/configs/train/custom.yaml
```
for multiple GPUs on one node run
```
torchrun --nproc_per_node=#num_gpus dinov2/train/train.py --config-file dinov2/configs/train/custom.yaml
```
## Citing DinoBloom

If you find this repository useful, please consider citing our work:

```
tbd
```
