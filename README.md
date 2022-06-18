# Global Vision Context Transformer (GC ViT)

This repository is the official PyTorch implementation of Global Vision Context
Transformers.


## Updates
***06/17/2022***

ImageNet-1K classification code for GC ViT released.

## Introduction

**GC ViT Transformer** leverages global context self-attention modules, joint with local self-attention, to effectively yet efficiently model both long and short-range spatial interactions, without the need for expensive 
operations such as computing attention masks or shifting local windows.

GC ViT  achieves state-of-the-art results across image classification, object detection and semantic segmentation tasks. On ImageNet-1K dataset for classification, the base, small and tiny variants of GC ViT with `28`M, `51`M and `90`M parameters achieve `83.2`, `83.9` and `84.4` Top-1 accuracy, respectively, surpassing comparably-sized prior art such as CNN-based ConvNeXt and ViT-based Swin Transformer by a large margin. Pre-trained GC ViT backbones in downstream tasks of object detection, instance segmentation, 
and semantic segmentation using MS COCO and ADE20K datasets outperform prior work consistently, sometimes by large margins.

![teaser](./assets/attention.png)

# Results on ImageNet

**ImageNet-1K Pretrained Models**

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1</th>
    <th>Resolution</th>
    <th>#Params</th>
    <th>FLOPs</th>
    <th>Performance Summary</th>
    <th>Tensorboard</th>
    <th>Download </th>
  </tr>
<tr>
    <td>GC ViT-T</td>
    <td>83.2</td>
    <td>224x224</td>
    <td>28</td>
    <td>4.7</td>
    <td><a href="https://drive.google.com/file/d/1FkuzQPMS5hIKrNtvP0KUpWENWIUmUCgD/view?usp=sharing">summary</a></td>
    <td><a href="https://tensorboard.dev/experiment/z3JpGy29T0awoDlRofeLPA">tensorboard</a></td>
    <td><a href="https://add_to_model">model</a></td>
</tr>

<tr>
    <td>GC ViT-S</td>
    <td>83.9</td>
    <td>224x224</td>
    <td>51</td>
    <td>8.5</td>
    <td><a href="https://drive.google.com/file/d/1--l6JycRAOu0HKJLJzpSe3gT7SyLpAkr/view?usp=sharing">summary</a></td>
    <td><a href="https://tensorboard.dev/experiment/YRDDGcYwTrWddlrp0zzX8w">tensorboard</a></td>
    <td><a href="https://add_to_model">model</a></td>
</tr>

<tr>
    <td>GC ViT-B</td>
    <td>84.4</td>
    <td>224x224</td>
    <td>90</td>
    <td>14.8</td>
    <td><a href="https://drive.google.com/file/d/167gxKLksnyQ9YMH6reGInKOiWAWWZuQ0/view?usp=sharing">summary</a></td>
    <td><a href="https://tensorboard.dev/experiment/qaLWmWiLRBOcBNgeUanLMg">tensorboard</a></td>
    <td><a href="https://add_to_model">model</a></td>
</tr>

</table>

