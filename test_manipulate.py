from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
import matplotlib.pyplot as plt
from torchvision.utils import *

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

# Bilder Laden
data = ImageDataset('/home/yv312705/Code/diffusion_autoenc/datasets/test_autoenc', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = data[0]['img'][None]

# Encoder
cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(xT[0].permute(1, 2, 0).cpu())

# Auswahl der zu manipulierenden Attribute
print(MriAttrDataset.id_to_cls)

# Eingabe des zu manipulierenden Attributs
cls_id = MriAttrDataset.cls_to_id['t1_weighted']


cond2 = cls_model.normalize(cond)
cond2 = cond2 + 0.3 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
cond2 = cls_model.denormalize(cond2)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
img = model.render(xT, cond2, T=100)
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[0].set_title('Original Image', fontsize= 14)
ax[1].imshow(img[0].permute(1, 2, 0).cpu())
ax[1].set_title('Manipulated Image', fontsize= 14)

antwort = input("Möchten Sie die Figur speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "classifier_t2.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")

    plt.show()