# GC ViT - Object Detection
This repository is the official PyTorch implementation of Global Context Vision Transformers for object detection using MS COCO dataset.

## Requirements
The dependencies can be installed by running:

```bash
pip install -r requirements.txt
```


## Benchmarks

The expected performance of models that use GC ViT as a backbone is listed below:

| Backbone | Head | #Params(M) | FLOPs(G) | mAP | Mask mAP|
|---|---|---|---|---|---|
| GC ViT-T | Mask R-CNN | 48 | 291 | 47.9 | 43.2 |
| GC ViT-T | Cascade Mask R-CNN | 85 | 770 | 51.6 | 44.6 |
| GC ViT-S | Cascade Mask R-CNN | 108 | 866 | 52.4 | 45.4 |
| GC ViT-B | Cascade Mask R-CNN | 146 | 1018 | 52.9 | 45.8 |


