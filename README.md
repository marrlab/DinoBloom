
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

## Sample Notebook
We provide a sample [google colab notebook](https://colab.research.google.com/drive/1fjArdu28G5_C9Hq2Qe08bSJGu2Bk2rni#scrollTo=AiRK-3cd9Uyh) that shows feature extraction and how to do PCA visualization.

## Citing DinoBloom

If you find this repository useful, please consider citing our work:

```
@misc{koch2024dinobloom,
      title={DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology}, 
      author={Valentin Koch and Sophia J. Wagner and Salome Kazeminia and Ece Sancar and Matthias Hehr and Julia Schnabel and Tingying Peng and Carsten Marr},
      year={2024},
      eprint={2404.05022},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Overview of pu

# Blood Cell Datasets

| Dataset                  | Modality    | #Patient Labels | Patient Labels                                                                                         | Cell/Image Labels                                                                                                          | Comment                                        | Source Link                                                                                               | Publication Link                                                                                         |
|--------------------------|-------------|-----------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| BMC                      | Bone marrow | 171,373         | -                                                                                                      | 21: ABE (Abnormal eosinophils), ART (Artefacts), BAS (Basophils), BLA (Blasts), EBO (Erythroblasts), EOS (Eosinophils), FGC (Faggot cells), HAC (Hairy cells), KSC (Sudge cells), LYI (Immature lymphocytes), LYT (Lymphocytes), MMZ (Metamyelocytes), MON (Monocytes), MYB (Myelocytes), NGB (Band neutrophils), NGS (Segmented neutrophils), NIF (Not identifiable), OTH (Other cells), PEB (Proerythoblasts), PLM (Plasma cells), PMO (Promyelocytes) | -                                              | [Link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770)                      | [Link](https://ashpublications.org/blood/article/138/20/1917/477932/Highly-accurate-differentiation-of-bone-marrow) |
| AML Hehr                 | Blood       | 101,949         | 4: PML::RARA, NPM1, CBFB::MYH11, RUNX1::RUNX1T1                                                        | -                                                                                                                          | -                                              | [Link](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/)                  | [Link](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000187) |
| AML Matek                | Blood       | 18,365          | -                                                                                                      | 15: BAS (Basophil), EBO (Erythroblast), EOS (Eosinophil), KSC (Smudge cell), LYA (Lymphocyte (atypical)), LYT (Lymphocyte (typical)), MMZ (Metamyelocyte), MOB (Monoblast), MON (Monocyte), MYB (Myelocyte), MYO (Myeloblast), NGB (Neutrophil (band)), NGS (Neutrophil (segmented)), PMB (Promyelocyte (bilobled)), PMO (Promyelocyte) | -                                              | [Link](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)                            | [Link](https://doi.org/10.7937/tcia.2019.36f5o9ld) |
| Acevedo                  | Blood       | 17,092          | -                                                                                                      | 10: basophil, eosinophil, erythroblast, lymphocyte_typical, metamyelocyte, monocyte, myelocyte, neutrophil_band, neutrophil_segmented, promyelocyte | -                                              | [Link](https://data.mendeley.com/datasets/snkd93bnjr/1)                                                   | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0169260719303578?via%3Dihub) |
| Raabin WBC               | Blood       | 10,175          | -                                                                                                      | 5: Eosinophil, Lymphocyte, Monocyte, Neutrophil, Basophil                                                                   | -                                              | [Link](https://raabindata.com/free-data/)                                                                | [Link](https://www.nature.com/articles/s41598-021-04426-x) |
| NuClick                  | Blood       | 2,933           | -                                                                                                      | -                                                                                                                          | Segmentation                                   | [Link](https://github.com/navidstuv/NuClick)                                                              | [Link](https://arxiv.org/pdf/2005.14511.pdf) |
| Warty pig                | Blood       | 2,871           | -                                                                                                      | 4: Basophil, Eosinophil, Monocyte, Neutrophil                                                                             | 667 raw images, 1464 augmented images, and 1408 cropped, classified images | [Link](https://drive.google.com/drive/folders/1CsDoL448kvAtFVd5jowVJGKjFLv3qjz4)                           | [Link](https://ieee-dataport.org/documents/dataset-machine-learning-based-classification-white-blood-cells-juvenile-visayan-warty-pig) |
| LISC                     | Blood       | 2,263           | -                                                                                                      | 5: Basophil, Eosinophil, Monocyte, Neutrophil, Lymphocyte                                                                  | segmentation                                              | [Link](http://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm)                               | Rezatofighi, S. H. & Soltanian-Zadeh, H. Automatic recognition of five types of white blood cells in peripheral blood. Comput. Med. Imaging Graph 35, 333–343 (2011). |
| KRD-WBC                  | Blood       | 601             | -                                                                                                      | 5: Eosinophil, Lymphocyte, Monocyte, Neutrophil, Basophil                                                                  | Segmentation                                   | [Link](https://data.mendeley.com/datasets/jzdj6h7gms/2)                                                   | Taha, Haval; Alizadeh, Fattah ; Mohammad, Nawsherwan  (2023), “Creating a white blood cell dataset for segmentation”, Mendeley Data, V2, doi: 10.17632/jzdj6h7gms.2 |
| SSL Seg                  | Blood       | 400             | -                                                                                                      | -                                                                                                                          | Segmentation                                   | [Link](https://github.com/zxaoyou/segmentation_WBC)                                                       | Zheng, X., Wang, Y., Wang, G. & Liu, J. Fast and robust segmentation of white blood cell images by self-supervised learning. Micron 107, 55–71 (2018). |
| BCCD                     | Blood       | 364             | -                                                                                                      | 3: WBC, RBC, Platelet                                                                                                      | detection                                              | [Link](https://www.kaggle.com/datasets/konstantinazov/bccd-dataset)                                        | Mohamed, M., Far, B. & Guaily, A. An efficient technique for white blood cells nuclei automatic segmentation. in 2012 IEEE International Conference on Systems, Man, and Cybernetics (SMC) 220–225 (2012). |
| Aslan                    | Blood       | 100             | -                                                                                                      | 2: WBC, RBC                                                                                                                          | detection                                              | [Link](https://github.com/draaslan/blood-cell-detection-dataset?tab=MIT-1-ov-file)                        | -                                                                                                             |
| Raabin Leukemia | Blood       | ?        | 4: Acute Lymphoblastic Leukemia, Acute Myeloblastic Leukemia, Chronic Lymphocytic Leukemia, Chronic Myelogenous Leukemia | -                                                                                                                          | -                                              | [Link](https://raabindata.com/free-data/)                                                                | -                                                                                                             |
| APL_AML                  | Blood       | 25,915          | 2: APL / AML non APL                                                                                   | Artifact, Band neutrophils, Basophil, Blast (no lineage spec), Eosinophils, Erythroblast, Giant thrombocyte, Lymphocyte, Lymphocyte (variant), Metamyelocyte, Monocyte, Myelocyte, Plasma cells, Prolymphocyte, Promonocyte, Promyelocyte, Segmented neutrophils, Smudge cells, Thrombocyte aggregation, Unidentified, Young Unidentified | -                                              | [Link](https://www.kaggle.com/datasets/eugeneshenderov/acute-promyelocytic-leukemia-apl/data)             | [Link](https://pubmed.ncbi.nlm.nih.gov/33990660/) |
| White-Blood-Cell-dataset | Blood       | 376             | -                                                                                                      | -                                                                                                                          | Segmentation                                   | [Link](https://github.com/arbackes/White-Blood-Cell-dataset)                                              | Mohamed, M.M.A., Far, B.H.: An enhanced threshold based technique for white blood cells nuclei automatic segmentation. In: Healthcom, pp. 202–207. IEEE (2012)|

