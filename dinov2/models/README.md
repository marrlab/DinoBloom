Changes for vision mamba:

* changed precision to full
* separate dino and ibot head
* model checkpoint for patch size 16
  * changed patch size to 16
  * changed local crop size from 98 to 96
* uncommented the assertion for input size in timm's transformer projection layer for local crops
* set `if_rope=False` from true
* set `fused_add_norm=False` from true
* removed `self.fsdp_synchronize_streams()` from `ssl_meta_arch.py`

todo

* [x] debug training without loading checkpoint
* [ ] make training compatible with checkpoint (rope embedding + abs pos emb)
* [ ] interpolate rotary embedding (need to change size)
* [ ] interpolate abs embedding (given)
* [ ] enable distributed training
* [ ] include nicer in ssl_meta_arch
* [ ] remove argparsing
* [ ] checkout `if self.training and self.sample_drop_ratio > 0.0:` for nested attention bock training in `block.py`
