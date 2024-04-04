# DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology
<img src="media/logo.png" title="Cutie" width="500" /> 
Repository of [DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology](arxiv.org) that uses [DINOv2](https://arxiv.org/abs/2304.07193) and is adaptet from their [oringinal Github repository](https://github.com/facebookresearch/dinov2/tree/main/dinov2).

<img src="media/overview.png" title="Overview"  /> 


DinoBloom is a model family (ViTs) trained on a large cohort of 13 diverse publicly available datasets of single cells in peripheral blood and bone marrow. The trained models in the can be downloaded on [zenodo](zenodolink.com) in the variants DinoBloom-S, DinoBloom-B, DinoBloom-L and DinoBloom-G.

| Model         | Feature dim | #params |
|---------------|-------------|---------|
| DinoBloom-S   | 384         | 22M     |
| DinoBloom-B   | 768         | 86M     |
| DinoBloom-L   | 1024        | 304M    |
| DinoBloom-G   | 1536        | 1136M   |

To train the model: 


## Citing DinoBloom

If you find this repository useful, please consider giving a star ⭐️ and citation :t-rex::

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
