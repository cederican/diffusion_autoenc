import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image




def find_nifti_files(input_dir, output_dir):
    nifti_files = []
    index = 00000

    for subdir, dirs, files in sorted(os.walk(input_dir)):
        for file in sorted(files):
            if file.endswith('.nii.gz') and not file.endswith('seg.nii.gz'):
                nifti_files.append(os.path.join(subdir, file))    
    

    # Iterate through the list of files and save each slice as a separate .png file
    for file_path in tqdm(nifti_files[:50]):  # loop through only the first 10 files
        # Load the .nifti file
        nii_data = nib.load(file_path).get_fdata()
        nii_data= np.clip((nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data)), 0, 1)
        nii_data_header = nib.load(file_path).header
        print(nii_data_header)
        print(file_path)

        # Get the metadata
        file_name = os.path.basename(file_path)
        name , ext = file_name.rsplit(".", 1)
        path_split = os.path.split(file_path)[0].split(os.sep)
        orientation_name = path_split[-1]

        # Loop through the slices and save each one as a separate .png file
        for i in range(nii_data.shape[2]):
            slice_data = nii_data[:,:,i]

            index_str = "{:05d}".format(index)
            slice_name = f"{index_str}.png"
            slice_path = os.path.join(output_dir, slice_name)
            
            # then crop the image horizontally to get two images out of it 
            # Öffnen des Originalbildes
            image = Image.fromarray((slice_data * 255).astype(np.uint8))

            #graustufenbild
            image = image.convert('L')

            # Definieren der Ausschnittsbreite und -höhe
            width, height = image.size
            left = 0
            top = 0
            right = width
            bottom = height // 2

            # Ausschneiden des linken Bildes
            top_image = image.crop((left, top, right, bottom))

            # Speichern des linken Bildes als PNG
            top_image.save(slice_path)

            index += 1
            index_str = "{:05d}".format(index)
            slice_name = f"{index_str}.png"
            slice_path = os.path.join(output_dir, slice_name)

            # Ausschneiden des rechten Bildes
            top = height // 2
            bottom = height
            bottom_image = image.crop((left, top, right, bottom))

            # Speichern des rechten Bildes als PNG
            bottom_image.save(slice_path)

            index += 1


input_dir = '/home/yv312705/Code/diffusion_autoenc/datasets/Unet_Darius/train'
output_dir = '/home/yv312705/Code/diffusion_autoenc/datasets/Unet_Darius_png'

find_nifti_files(input_dir, output_dir)


