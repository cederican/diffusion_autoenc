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

torch.set_printoptions(threshold=torch.inf)

print(plt.get_backend())

# Backend auf "agg" ändern
plt.switch_backend('agg')

# Neues Backend anzeigen
print(plt.get_backend())

# Diffusion Autoencoder Laden
device = 'cuda:0'
conf = mri_autoenc()
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

# Classifier Laden
cls_conf = mri_autoenc_cls()
cls_model = ClsModel(cls_conf)
state = torch.load(f'checkpoints/{cls_conf.name}/last.ckpt',
                    map_location='cpu')
print('latent step:', state['global_step'])
cls_model.load_state_dict(state['state_dict'], strict=False)
cls_model.to(device)

index = 00000

# ------------- Originale Bilder Laden -------------
data = ImageDataset('/home/yv312705/Code/diffusion_autoenc/eval_metrics/realPD/PD', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False, sort_names=True)
for p in range(2):
    batch = data[p]['img'][None]
    ori = (batch + 1) / 2

    # -------------------- Encoder ------------------
    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)

    # --------------------- Classifier Test ----------
    cond = cls_model.normalize(cond)
    pred = cls_model.classifier.forward(cond)
    print('pred:', pred)
    cond = cls_model.denormalize(cond)

    # ----------- Auswahl der zu manipulierenden Attribute -----------
    print(MriAttrDataset.id_to_cls)

    # ----------- Eingabe des zu manipulierenden Attributs -----------
    cls_id = MriAttrDataset.cls_to_id['cor_pd_fs']

    cond_class = cls_model.normalize(cond)
    cond_class = cond_class + 0.3 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
    prediction = cls_model.classifier.forward(cond_class)
    cond_class = cls_model.denormalize(cond_class)

    img = model.render(xT, cond_class, T=100)
    img = (img *255).byte()
    img = np.array(img[0,0].cpu())

    ori = (ori *255).byte()
    ori = np.array(ori[0,0].cpu())

    index_str = "{:05d}".format(index)
    slice_name = f"{index_str}.png"
    slice_name1 = f"{index_str}_ori.png"
    output_path = os.path.join('/home/yv312705/Code/diffusion_autoenc/testing', slice_name)
    output_path1 = os.path.join('/home/yv312705/Code/diffusion_autoenc/testing', slice_name1)
    cv2.imwrite(output_path, img)
    cv2.imwrite(output_path1, ori)


    index += 1