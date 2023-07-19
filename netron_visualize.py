from model.unet import BeatGANsUNetModel, BeatGANsUNetConfig
from model.unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel
from templates import *

import netron
import torch

# ----------------------- function to visualize the DiffAE net -------------------
model = BeatGANsAutoencModel(BeatGANsAutoencConfig)
checkpoint = torch.load('/home/yv312705/Code/diffusion_autoenc/checkpoints/mri_autoenc/last.ckpt')
model.load_state_dict(checkpoint['state_dict'], strict=False)

input_example = torch.randn(1, 3, 128, 128)
torch.onnx.export(model, input_example, '/home/yv312705/Code/diffusion_autoenc/netron_model.onnx', input_names=['Original MRI Image 128x128'], output_names=['Reconstructed MRI Image 128x128'])

netron.start('/home/yv312705/Code/diffusion_autoenc/netron_model.onnx')