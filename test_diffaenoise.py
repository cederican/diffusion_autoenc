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
pred_list = model.render(xT, cond, T=500)

x_500 = (pred_list[0]["sample"] + 1) /2
x_200 = (pred_list[299]["sample"] + 1) /2
x_100 = (pred_list[399]["sample"] + 1) /2
x_50 = (pred_list[449]["sample"] + 1) /2
x_20 =  (pred_list[479]["sample"] + 1) /2
x_10 =  (pred_list[489]["sample"] + 1) /2
x_5 =  (pred_list[494]["sample"] + 1) /2
x_2 =  (pred_list[497]["sample"] + 1) /2
x_1 =  (pred_list[498]["sample"] + 1) /2
x_0 =  (pred_list[499]["sample"] + 1) /2

fig, ax = plt.subplots(1, 10, figsize=(14, 8))

ax[0].imshow(x_500[0].permute(1, 2, 0).cpu())
ax[0].set_title('X_500', fontsize= 14)
ax[1].imshow(x_200[0].permute(1, 2, 0).cpu())
ax[1].set_title('X_200', fontsize= 14)
ax[2].imshow(x_100[0].permute(1, 2, 0).cpu())
ax[2].set_title('X_100', fontsize= 14)
ax[3].imshow(x_50[0].permute(1, 2, 0).cpu())
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


# ssim
'''
pred = pred.to(device)
ori = ori.to(device)
ssim = SSIM()
score = ssim(ori, pred)
ssim_score = "SSIM Score: " + str(score.item())
print(ssim_score)
'''

# Differenz der Bilder
#diff_tensor = torch.abs(ori[0].permute(1, 2, 0).cpu() - pred[0].permute(1, 2, 0).cpu())

'''
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
'''
antwort = input("Möchten Sie die Figur speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_seven/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "autoencoder8M20K_T1000_noise.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")
    
