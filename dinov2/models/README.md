Changes for vision mamba:

* model checkpoint for patch size 16
* * changed patch size to 16
  * changed local crop size from 98 to 96
* uncommented the assertion for input size in timm's transformer projection layer for local crops
* set `if_rope=False` from true as rope embeddings have different shapes for different sizes
* removed `self.fsdp_synchronize_streams()` from `ssl_meta_arch.py`

todo

* [X] debug training without loading checkpoint
* [X] make training compatible with checkpoint (+ abs pos emb)
* [X] enable distributed training
* [ ] interpolate rotary embedding (need to change size)
* [X] include nicer in ssl_meta_arch
* [X] remove argparsing
* [ ] checkout `if self.training and self.sample_drop_ratio > 0.0:` for nested attention bock training in `block.py`
