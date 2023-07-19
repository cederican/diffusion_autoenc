from templates import *
import numpy as np
import matplotlib.pyplot as plt
from templates_cls import *
from experiment_classifier import ClsModel
from PIL import Image


device = 'cuda:0'
conf = mri_autoenc()
# print(conf.name)
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

alpha = torch.tensor(np.linspace(0, 1, 10, dtype=np.float32)).to(cond.device)
intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]
for i in range(len(intp)):
    test_cond = cls_model.normalize(intp[i])
    prediction= cls_model.classifier.forward(test_cond)
    print('pred:', prediction)
    test_cond = cls_model.denormalize(test_cond)

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

pred = model.render(intp_x, intp, T=100)

fig, ax = plt.subplots(1, 10, figsize=(5*20, 5))
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

frames[0].save('/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/interpolate.gif', format='GIF', save_all=True, append_images=frames[1:], duration=150, loop=0)

#schicker plot bilderreihe
breite, höhe = frames[0].size
ausgabe = Image.new('RGBA', (breite * len(frames), höhe))
for index, bild in enumerate(frames):
    ausgabe.paste(bild, (index * breite, 0))
ausgabe.save('/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/niceplot.png')


antwort = input("Möchten Sie die Figur speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "interpolate.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")

