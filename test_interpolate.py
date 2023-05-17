from templates import *
import numpy as np
import matplotlib.pyplot as plt


device = 'cuda:0'
conf = mri_autoenc()
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

data = ImageDataset('/home/yv312705/Code/diffusion_autoenc/datasets/test_interpolate', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = torch.stack([
    data[0]['img'],
    data[1]['img'],
])

plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5)

# Encoder

cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(xT[0].permute(1, 2, 0).cpu())

# Interpolation

alpha = torch.tensor(np.linspace(0, 1, 40, dtype=np.float32)).to(cond.device)
intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

theta = torch.arccos(cos(xT[0], xT[1]))
x_shape = xT[0].shape
intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
intp_x = intp_x.view(-1, *x_shape)

pred = model.render(intp_x, intp, T=1000)

fig, ax = plt.subplots(1, 40, figsize=(5*20, 5))
for i in range(len(alpha)):
    ax[i].imshow(pred[i].permute(1, 2, 0).cpu())

fig.suptitle('Interpolation Testing', fontsize= 20)

# Gif erstellen
frames = []

for i in range(len(alpha)):
    numpy_array = (pred[i].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    image = Image.fromarray(numpy_array)
    img_resized = image.resize((image.size[0]*2, image.size[1]*2))
    frames.append(img_resized)

frames[0].save('/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_two/interpolate.gif', format='GIF', save_all=True, append_images=frames[1:], duration=150, loop=0)



antwort = input("Möchten Sie die Figur speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_two/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "interpolate4M14K_T1000_20.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")

