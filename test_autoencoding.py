from templates import *
import matplotlib.pyplot as plt
from ssim import *
from PIL import Image

device = 'cuda:1'
conf = mri_autoenc()
#print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)


data = ImageDataset('/home/yv312705/Code/diffusion_autoenc/datasets/test_autoenc', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = data[0]['img'][None]

# original
ori = (batch + 1) / 2

# Encoder
cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

# Decoder
pred = model.render(xT, cond, T=1000)

# ssim
pred = pred.to(device)
ori = ori.to(device)
ssim = SSIM()
score = ssim(ori, pred)
ssim_score = "SSIM Score: " + str(score.item())
print(ssim_score)

# Differenz der Bilder
diff_tensor = torch.abs(ori[0].permute(1, 2, 0).cpu() - pred[0].permute(1, 2, 0).cpu())


# plot evaluation
fig, ax = plt.subplots(2, 3, figsize=(14, 8))

ax[0, 0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[0, 0].set_title('Original Image', fontsize= 14)
ax[0, 1].imshow(xT[0].permute(1, 2, 0).cpu())
ax[0, 1].set_title('Encoded Image', fontsize= 14)
ax[0, 2].spines['top'].set_visible(False)
ax[0, 2].spines['right'].set_visible(False)
ax[0, 2].spines['bottom'].set_visible(False)
ax[0, 2].spines['left'].set_visible(False)
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])
ax[1, 0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1, 0].set_title('Original Image', fontsize= 14)
ax[1, 1].imshow(pred[0].permute(1, 2, 0).cpu())
ax[1, 1].set_title('Decoded Image', fontsize= 14)
# range des diff bildes wichtig, sensitivität so einstellen mit original max value. vmin 0 für Gleichheit
ax[1, 2].imshow(diff_tensor.sum(2), cmap='jet', vmin=0, vmax=ori[0].permute(1, 2, 0).cpu().max())
ax[1, 2].set_title('Difference Original/Decoded', fontsize= 14)

fig.suptitle('Autoencoder Testing', fontsize= 17)
fig.text(0.85, 0.7, ssim_score, ha="center", va="center", fontsize=14)
fig.tight_layout()
fig.colorbar(ax[1, 2].imshow(diff_tensor.sum(2), cmap='jet', vmin=0, vmax=ori[0].permute(1, 2, 0).cpu().max()))
fig.subplots_adjust(hspace=0.3, wspace=0.1)

antwort = input("Möchten Sie die Figur speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_six/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "autoencoder12M9K_T1000.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")
    



