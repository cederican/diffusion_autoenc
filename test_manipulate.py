from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
import matplotlib.pyplot as plt
from torchvision.utils import *
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
import torchvision.transforms as transforms


# Diffusion Autoencoder Laden
device = 'cuda:1'
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


# Classifier mit ROC Plot testen #########################################
test_dir = ImageDataset('/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
test_size = test_dir.__len__()
test_data_dir = '/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier/'

subdirs = [subdir for subdir in sorted(os.listdir(test_data_dir)) if os.path.isdir(os.path.join(test_data_dir, subdir))]
label_map = {subdir: i for i, subdir in enumerate(subdirs)}

labels = []

y_predictedlabel = []
y_truelabel = []

for subdir in subdirs:
    subdir_path = os.path.join(test_data_dir, subdir)
    for filename in os.listdir(subdir_path):
        labels.append(label_map[subdir])

for i in range (0, test_size):
    test_batch = test_dir[i]['img'][None]
    cond = model.encode(test_batch.to(device))
    cond = cls_model.normalize(cond)
    pred = cls_model.classifier.forward(cond)
    #print(pred)

    _ , pred = torch.max(pred, dim=1)
    pred = pred.item()

    # ROC Metrics
    y_predictedlabel.append(pred)
    y_truelabel.append(labels[i])

fpr, tpr, thresholds = roc_curve(y_truelabel, y_predictedlabel)

auc_score = "AUC Score: " + str(auc(fpr, tpr).item())

print('AUC score:', auc_score)

# Plot the ROC curve
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.title(f'ROC Curve / Number of testdata: {test_size}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.text(0.5, 0.5, auc_score, ha="center", va="center", fontsize=14)

antwort = input("Möchten Sie die Figur des ROC Plots speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_six/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "ROC_plot.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")
######################################################################

    
# Bilder Laden
data = ImageDataset('/home/yv312705/Code/diffusion_autoenc/datasets/test_autoenc', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = data[0]['img'][None]

# Encoder
cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

# teste ob classifizierer funktioniert
cond = cls_model.normalize(cond)
pred = cls_model.classifier.forward(cond)
print('pred:', pred)
cond = cls_model.denormalize(cond)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(xT[0].permute(1, 2, 0).cpu())

# Auswahl der zu manipulierenden Attribute
print(MriAttrDataset.id_to_cls)

# Eingabe des zu manipulierenden Attributs
cls_id = MriAttrDataset.cls_to_id['cor_pd_fs']

images = []

stepsize = 0.015

for j in range(20):
    cond_class = cls_model.normalize(cond)
    cond_class = cond_class + stepsize * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)

    prediction = cls_model.classifier.forward(cond_class)
    print('pred:', prediction)

    cond_class = cls_model.denormalize(cond_class)
    img = model.render(xT, cond_class, T=100)
    cond_class = 0
    images.append(img)

    if stepsize == 0.3:
        break
    stepsize += 0.015


fig, ax = plt.subplots(1, 20, figsize=(5*20, 5))

for i in range(20):
    if i == 0:
        ori = (batch + 1) / 2
        ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
        ax[0].set_title('Original Image pdw', fontsize= 10)
    else:     
        ax[i].imshow(images[i][0].permute(1, 2, 0).cpu())

fig.suptitle(f'Manipulate Mode Testing', fontsize= 17)

# Gif erstellen
frames = []

for i in range(20):
    numpy_array = (images[i][0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    image = Image.fromarray(numpy_array)
    img_resized = image.resize((image.size[0]*2, image.size[1]*2))
    frames.append(img_resized)

frames[0].save('/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_six/pd2.gif', format='GIF', save_all=True, append_images=frames[1:], duration=150, loop=0)



antwort = input("Möchten Sie die Figur der Manipulation speichern? (ja/nein)")

# Wenn die Antwort "Ja" lautet, speichern Sie die Figur ab
if antwort.lower() == "ja":

    pfad = "/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_six/"

    if not os.path.exists(pfad):
        os.makedirs(pfad)

    plt.savefig(pfad + "classifier12M9K_pd_T1002.png")
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")
