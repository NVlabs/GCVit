import torch, time
from models.gc_vit_high_res import gc_vit_xtiny
import timm
import pdb
resolution = 1024
model = timm.create_model(
    model_name='gc_vit_xtiny',
    resolution=resolution,
    exportable=True)

in_size = (1, 3, resolution, resolution)
model = model.cuda()
imgs = torch.randn(in_size, device="cuda")
torch.onnx.export(model, imgs, "./gc_vit.onnx", input_names=['input'], verbose=False, output_names=['output'], export_params=True)
