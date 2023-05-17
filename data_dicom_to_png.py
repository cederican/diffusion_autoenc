import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image
import pydicom as dicom

index = 00000

output_dir = '/home/yv312705/Code/diffusion_autoenc/datasets/testdata/testexport/'

# Open the DICOM file
filename = '/home/yv312705/Code/diffusion_autoenc/datasets/testdata/Bavdm_Proband1/Mrhu - 726502493/IM-0001-0261.dcm'
ds = dicom.dcmread(filename)

# Extract the pixel array
pixel_array = ds.pixel_array.astype('float32')

# Normalize the pixel values to between 0 and 1
pixel_array /= np.max(pixel_array)

# Loop through each slice in the DICOM file and save it as a PNG image
for i, image in enumerate(pixel_array):

    index_str = "{:05d}".format(index)
    slice_name = f"{index_str}.png"
    slice_path = os.path.join(output_dir, slice_name)

    # Convert the numpy array to a PIL image
    img = Image.fromarray(np.uint8(image*255))

    cropped_image = img.crop((0, 100, img.width, img.height - 100))

    # Definieren der Ausschnittsbreite und -höhe
    width, height = cropped_image.size
    left = 0
    top = 0
    right = width // 2
    bottom = height

            # Ausschneiden des linken Bildes
    left_image = cropped_image.crop((left, top, right, bottom))

            # Speichern des linken Bildes als PNG
    left_image.save(slice_path)

    index += 1
    index_str = "{:05d}".format(index)
    slice_name = f"{index_str}.png"
    slice_path = os.path.join(output_dir, slice_name)

            # Ausschneiden des rechten Bildes
    right = width
    left = width // 2
    bottom_image = cropped_image.crop((left, top, right, bottom))

    bottom_image.save(slice_path)

    index += 1
        