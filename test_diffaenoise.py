from templates import *
import matplotlib.pyplot as plt
from ssim import *
from PIL import Image
import numpy as np


# ------------- script to create the denoising visualization -------------------
# -------------- important: change things in render() function ---------------

# ------- load model ---------------
device = 'cuda:1'
conf = mri_autoenc()
#print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

# ------------- load image ----------
data = ImageDataset('/home/yv312705/Code/diffusion_autoenc/datasets/test_autoenc', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = data[0]['img'][None]

# original
ori = (batch + 1) / 2

# Encoder
cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

# Decoder
pred_list = model.render(xT, cond, T=1000)

frames = []
frames_image = []

x_1000 = (pred_list[0]["sample"] + 1) /2
frames.append(x_1000)
x_900 = (pred_list[99]["sample"] + 1) /2
frames.append(x_900)
x_800 = (pred_list[199]["sample"] + 1) /2
frames.append(x_800)
x_700 = (pred_list[299]["sample"] + 1) /2
frames.append(x_700)
x_600 = (pred_list[399]["sample"] + 1) /2
frames.append(x_600)
x_500 = (pred_list[499]["sample"] + 1) /2
frames.append(x_500)
x_400 = (pred_list[599]["sample"] + 1) /2
frames.append(x_400)
x_300 = (pred_list[699]["sample"] + 1) /2
frames.append(x_300)
x_200 = (pred_list[799]["sample"] + 1) /2
frames.append(x_200)
x_100 = (pred_list[899]["sample"] + 1) /2
frames.append(x_100)
x_80 = (pred_list[919]["sample"] + 1) /2
frames.append(x_80)
x_60 = (pred_list[939]["sample"] + 1) /2
frames.append(x_60)
x_40 = (pred_list[959]["sample"] + 1) /2
frames.append(x_40)
x_20 =  (pred_list[979]["sample"] + 1) /2
frames.append(x_20)
x_10 =  (pred_list[989]["sample"] + 1) /2
frames.append(x_10)
x_5 =  (pred_list[994]["sample"] + 1) /2
frames.append(x_5)
x_2 =  (pred_list[997]["sample"] + 1) /2
frames.append(x_2)
x_1 =  (pred_list[998]["sample"] + 1) /2
frames.append(x_1)
x_0 =  (pred_list[999]["sample"] + 1) /2
frames.append(x_0)

for i in range(len(frames)):
    numpy_array = (frames[i][0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    image = Image.fromarray(numpy_array)
    frames_image.append(image)

#-------- schicker plot bilderreihe ---------
breite, höhe = frames_image[0].size
ausgabe = Image.new('RGBA', (breite * len(frames_image), höhe))
for index, bild in enumerate(frames_image):
    ausgabe.paste(bild, (index * breite, 0))
ausgabe.save('/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/autoencoder/denoise.png')

# ---------- old school plot ---------------- 
fig, ax = plt.subplots(1, 10, figsize=(14, 8))

ax[0].imshow(x_500[0].permute(1, 2, 0).cpu())
ax[0].set_title('X_500', fontsize= 14)
ax[1].imshow(x_200[0].permute(1, 2, 0).cpu())
ax[1].set_title('X_200', fontsize= 14)
ax[2].imshow(x_100[0].permute(1, 2, 0).cpu())
ax[2].set_title('X_100', fontsize= 14)
ax[3].imshow(x_60[0].permute(1, 2, 0).cpu())
ax[3].set_title('X_50', fontsize= 14)
ax[4].imshow(x_20[0].permute(1, 2, 0).cpu())
ax[4].set_title('X_20', fontsize= 14)
ax[5].imshow(x_10[0].permute(1, 2, 0).cpu())
ax[5].set_title('X_10', fontsize= 14)
ax[6].imshow(x_5[0].permute(1, 2, 0).cpu())
ax[6].set_title('X_5', fontsize= 14)
ax[7].imshow(x_2[0].permute(1, 2, 0).cpu())
ax[7].set_title('X_2', fontsize= 14)
ax[8].imshow(x_1[0].permute(1, 2, 0).cpu())
ax[8].set_title('X_1', fontsize= 14)
ax[9].imshow(x_0[0].permute(1, 2, 0).cpu())
ax[9].set_title('X_0', fontsize= 14)

# ----------- speichern ----------
antwort = input("Möchten Sie die Figur speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/autoencoder/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "autoencoder_denoise.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")
    
