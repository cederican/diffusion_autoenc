from templates import *
from templates_latent import *
import matplotlib.pyplot as plt

# --------------------- script to sample conditioned from latent space ------------------- 

# ------------------- load latent rained model ------------------
device = 'cuda:0'
conf = mri_autoenc_latent()
conf.T_eval = 100
conf.latent_T_eval = 100
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
print(model.load_state_dict(state['state_dict'], strict=False))
model.to(device)

# --------------- sample images --------------------
torch.manual_seed(1)
imgs = model.sample(8, device=device, T=1000, T_latent=200)

# ----------------- plot the generated samples -------------------
fig, ax = plt.subplots(2, 4, figsize=(4*5, 2*5))
ax = ax.flatten()
for i in range(len(imgs)):
    ax[i].imshow(imgs[i].cpu().permute([1, 2, 0]))
fig.suptitle('Latent sampling', fontsize= 18, fontweight='bold')

# ------------------------- speichern -------------------------
antwort = input("Möchten Sie die Figur speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/sampling/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "dpm_sample_T1000.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")

