from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
import matplotlib.pyplot as plt
from torchvision.utils import *
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np
import cv2 as cv2
import torch
import torchvision.transforms as transforms
from ssim import *

# ----------------- cluster spezifisch remote -------------------- 
torch.set_printoptions(threshold=torch.inf)
print(plt.get_backend())
plt.switch_backend('agg')
print(plt.get_backend())

# ---------------- Diffusion Autoencoder Laden ----------------
device = 'cuda:0'
conf = mri_autoenc()
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

# ------------------ Classifier Laden -------------------------
cls_conf = mri_autoenc_cls()
cls_model = ClsModel(cls_conf)
state = torch.load(f'checkpoints/{cls_conf.name}/last.ckpt',
                    map_location='cpu')
print('latent step:', state['global_step'])
cls_model.load_state_dict(state['state_dict'], strict=False)
cls_model.to(device)

# ---------------- Multiclass Classifier Evaluation -------------------
test_dir = ImageDataset('/home/yv312705/Code/diffusion_autoenc/eval_metrics/COR PD FS 04/fakeT1', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False, sort_names=True)
test_size = test_dir.__len__()

for i in range (0, test_size):
    test_batch = test_dir[i]['img'][None]
    cond = model.encode(test_batch.to(device))
    cond = cls_model.normalize(cond)
    pred = cls_model.classifier.forward(cond)
    print(pred)