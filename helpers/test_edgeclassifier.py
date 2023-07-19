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

# --------------------------- funktion um canny edge algorithmus herumzuprobieren ----------------------
# ------------------ not important -----------------------
torch.set_printoptions(threshold=torch.inf)

print(plt.get_backend())

# Backend auf "agg" ändern
plt.switch_backend('agg')

# Neues Backend anzeigen
print(plt.get_backend())

data_dir = '/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/edge_test'

stepsize = 0.005
num_steps = 200
stepsizes = []
images = []
edge_images = []

for image in sorted(os.listdir(data_dir)):
     image_path = os.path.join(data_dir, image)
     img = Image.open(image_path)
     img_array = np.array(img)
     images.append(img_array)


for i in range(num_steps):
     stepsizes.append(stepsize)
     stepsize+=0.005

original_edgeimg = images[0]
original_edgeimg = cv2.Canny(original_edgeimg, threshold1=100, threshold2=150)
cv2.imwrite('/home/yv312705/Code/diffusion_autoenc/eval_plots/testedges.png', original_edgeimg)


def edgeclassifier(images, original_edgeimg):

        original_edgetensor = torch.from_numpy(original_edgeimg).unsqueeze(0)
        diff_tensors = []

        # ---------- edge images syn ---------
        for i in range(num_steps):
            edge_image = cv2.Canny(images[i], threshold1=30, threshold2=90)
            edge_images.append(edge_image)

        # ------- Abs. Diff ----------
        for j in range(num_steps):
            edge_tensor = torch.from_numpy(edge_images[j]).unsqueeze(0)
            diff_tensor = torch.abs(edge_tensor.permute(1, 2, 0).cpu() - original_edgetensor.permute(1, 2, 0).cpu())
            diff_tensors.append(diff_tensor)

        return diff_tensors, edge_images

diff_tens, edge_images = edgeclassifier(images, original_edgeimg)

# ----------- compute the lowest difference between original and manip in edges -------------
counter_list = []
threshhold = 0.4

for diff_t in diff_tens:
    counter = 0
    for row in range(diff_t.shape[0]):
        if row < 5:
            continue
        
        if row >= diff_t.shape[0] - 5:
            continue
        
        for col in range(diff_t.shape[1]):
            pixel = diff_t[row, col]
            if pixel != 0:
                counter += 1

    counter_list.append(counter)

sorted_counter = sorted(enumerate(counter_list), key=lambda x: x[1])

for k in range(num_steps):
    smallest_index = sorted_counter[k][0]
    selected_stepsize = stepsizes[smallest_index]
    if selected_stepsize > threshhold:
        break

print(selected_stepsize)

# --------------- Big Plot of ori, edges, diff_edges and best image -------------------
fig, ax = plt.subplots(4, num_steps, figsize=(5*30, 10))

for i in range(num_steps):
        
        ax[0,i].imshow(images[i], cmap='gray')
        ax[0,i].axis('off')
        ax[0,i].set_title(str(i), fontsize= 10)
        ax[1,i].imshow(edge_images[i], cmap='gray')
        ax[1,i].axis('off')
        ax[1,i].set_title(str(i), fontsize= 10)
        ax[2,i].imshow(diff_tens[i].sum(2), cmap='gray', vmin=0, vmax=100)
        ax[2,i].axis('off')
        ax[2,i].set_title(str(i), fontsize= 10)
fig.suptitle(f'Manipulate Mode Testing', fontsize= 17)
#antwort = input("Möchten Sie die Figur des Big Plot speichern? (ja/nein)")
antwort = "ja"
if antwort.lower() == "ja":
    pfad = f'/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/test/'
    if not os.path.exists(pfad):
        os.makedirs(pfad)
    plt.savefig(pfad + 'edgetest.png')
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")


 # ----------------------- Small Plot of ori and best image ----------------------

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(images[0], cmap='gray')
ax[0].set_title('Original Image', fontsize= 10, fontweight='bold')
ax[1].imshow(images[smallest_index], cmap='gray')
ax[1].set_title('Manipulated Image', fontsize= 10, fontweight='bold')
fig.suptitle(f'Manipulation Mode', fontsize= 17, fontweight='bold')
#antwort = input("Möchten Sie die Figur des Small Plot speichern? (ja/nein)")
antwort = "ja"
if antwort.lower() == "ja":
    pfad = f'/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/test/'
    if not os.path.exists(pfad):
        os.makedirs(pfad)
    plt.savefig(pfad + 'small.png')
    print("Figur wurde gespeichert!")
else:
    print("Figur wurde nicht gespeichert.")

 # ------------------ Gif erstellen ----------------------
frames = []
for i in range(smallest_index):
    numpy_array = (images[i])
    image = Image.fromarray(numpy_array)
    img_resized = image.resize((image.size[0], image.size[1]))
    frames.append(img_resized)
frames[0].save(f'/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/test/gif.gif',format='GIF', save_all=True, append_images=frames[1:], duration=150, loop=0)

