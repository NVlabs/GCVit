import torch
import onnx
import onnxruntime as ort

onnx_model = onnx.load("./gc_vit.onnx")
onnx.checker.check_model(onnx_model)
x = torch.randn((1, 3, 1024, 1024))
ort_sess = ort.InferenceSession('gc_vit.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})
print(f'Predicted shape: "{outputs[0].shape}""')
