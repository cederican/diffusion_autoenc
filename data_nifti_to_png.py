import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image
import json
import random



nifti_files = []

def make_square(img):
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        diff = (width - height) // 2
        box = (diff, 0, width - diff, height)
    else:
        diff = (height - width) // 2
        box = (0, diff, width, height - diff)
    return img.crop(box)

def process_patient_dir(patient_dir):
    for study_dir in os.listdir(patient_dir):
        study_dir_path = os.path.join(patient_dir, study_dir)
        if os.path.isdir(study_dir_path) and study_dir.startswith('Study_'):
            # Suche nach dem target-Verzeichnis in den Serienverzeichnissen
            for series_dir in os.listdir(study_dir_path):
                series_dir_path = os.path.join(study_dir_path, series_dir)
                if os.path.isdir(series_dir_path) and series_dir.startswith('Series_'):
                    # Prüfe, ob der target-Ordner in dieser Serie ist
                    json_file_path = os.path.join(series_dir_path, 'body_parts.json')
                    with open(json_file_path, 'r') as f:
                        json_data = json.load(f)
                    target_dir = None
                    for key, value in json_data.items():
                        if value == 'knie':
                            target_dir = os.path.join(series_dir_path, key)
                            break
                    if target_dir is not None:
                        # Lese alle .nii.gz-Dateien im target-Verzeichnis ein
                        for file in os.listdir(target_dir):
                            if file.endswith('.nii.gz') and not file.endswith('seg.nii.gz'):
                                nifti_files.append(os.path.join(target_dir, file))




def find_nifti_files(input_dir, output_dir):

    index = 00000
    for network_dir in sorted(os.listdir(input_dir)):
        network_dir_path = os.path.join(input_dir, network_dir)
        if os.path.isdir(network_dir_path) and not network_dir.startswith('out'):
            for patient_dir in sorted(os.listdir(network_dir_path)):
                patient_dir_path = os.path.join(network_dir_path, patient_dir)
                if os.path.isdir(patient_dir_path) and patient_dir.startswith('Patient_'):
                    process_patient_dir(patient_dir_path)  
    

    # Iterate through the list of files and save each slice as a separate .png file
    for file_path in tqdm(nifti_files):  # loop through only the first 10 files
        # Load the .nifti file
        nii_data = nib.load(file_path).get_fdata()
        nii_data= np.clip((nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data)), 0, 1)
        nii_data_header = nib.load(file_path).header
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

            top_image = make_square(top_image)

            # Zufällige Drehung um 90 Grad
            angle = random.choice([0, 90, 180, 270])
            top_image = top_image.rotate(angle)

            # Zufällige Spiegelung
            if random.choice([True, False]):
                top_image = top_image.transpose(method=Image.FLIP_LEFT_RIGHT)  # horizontal
            else:
                top_image = top_image.transpose(method=Image.FLIP_TOP_BOTTOM)  # vertikal

            # Speichern des linken Bildes als PNG
            top_image.save(slice_path)

            top_image = top_image.transpose(method=Image.FLIP_TOP_BOTTOM) 
            index += 1
            index_str = "{:05d}".format(index)
            slice_name = f"{index_str}.png"
            slice_path = os.path.join(output_dir, slice_name)
            top_image.save(slice_path)


            index += 1
            index_str = "{:05d}".format(index)
            slice_name = f"{index_str}.png"
            slice_path = os.path.join(output_dir, slice_name)

            # Ausschneiden des rechten Bildes
            top = height // 2
            bottom = height
            bottom_image = image.crop((left, top, right, bottom))

            bottom_image = make_square(bottom_image)

            # Zufällige Drehung um 90 Grad
            angle = random.choice([0, 90, 180, 270])
            bottom_image = bottom_image.rotate(angle)

            # Zufällige Spiegelung
            if random.choice([True, False]):
                bottom_image = bottom_image.transpose(method=Image.FLIP_LEFT_RIGHT)  # horizontal
            else:
                bottom_image = bottom_image.transpose(method=Image.FLIP_TOP_BOTTOM)  # vertikal

            # Speichern des rechten Bildes als PNG
            bottom_image.save(slice_path)

            bottom_image = bottom_image.transpose(method=Image.FLIP_TOP_BOTTOM)
            index += 1
            index_str = "{:05d}".format(index)
            slice_name = f"{index_str}.png"
            slice_path = os.path.join(output_dir, slice_name)
            bottom_image.save(slice_path)


            index += 1


input_dir = '/home/yv312705/Code/diffusion_autoenc/datasets/Unet_Darius'
output_dir = '/home/yv312705/Code/diffusion_autoenc/datasets/Unet_Darius_test'

find_nifti_files(input_dir, output_dir)
