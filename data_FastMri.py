import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2
from tqdm import tqdm
from PIL import Image, ImageEnhance



def convert_dicom_to_png(input_folder, output_folder1, output_folder2, sequence_name1, sequence_name2):

    index1 = 00000
    index2 = 00000
    


    subfolders = [f for f in sorted(os.listdir(input_folder)) if os.path.isdir(os.path.join(input_folder, f))]

    for folder_name in tqdm(subfolders[:200]):
        folder_path = os.path.join(input_folder, folder_name)
        study_folders = [f for f in sorted(os.listdir(folder_path)) if os.path.isdir(os.path.join(folder_path, f))]
        
        for folder_name2 in study_folders:
            folder_path2 = os.path.join(folder_path, folder_name2)
            sequence_folders = [f for f in sorted(os.listdir(folder_path2)) if os.path.isdir(os.path.join(folder_path2, f))]

            for dicom_files in sequence_folders:
                dicom_path = os.path.join(folder_path2, dicom_files)

                files = sorted(os.listdir(dicom_path))

                # Suche nach den Exec-Dateien im aktuellen Ordner
                exec_files = [f for f in files if not os.path.isdir(os.path.join(folder_path2, f)) and not f.endswith('.jpg')]

                # Durchlaufe die Exec-Dateien
                for exec_file in exec_files:
                    exec_file_path = os.path.join(dicom_path, exec_file)

                      
                    ds = pydicom.dcmread(exec_file_path)
                    
                    if hasattr(ds, 'pixel_array'):
                    
                        # Überprüfe, ob die aktuelle DICOM-Datei die gewünschte Bildsequenz enthält
                        if ds.SeriesDescription == sequence_name1:
                            # Extrahiere das DICOM-Bild als Numpy-Array
                            pixel_array = ds.pixel_array.astype('float32')
                            pixel_array /= np.max(pixel_array)
                            img = Image.fromarray(np.uint8(pixel_array*255))
                            # Speichere das DICOM-Bild als PNG-Datei im Ausgabeverzeichnis
                            index_str1 = "{:05d}".format(index1)
                            slice_name1 = f"{index_str1}.png"
                            output_path = os.path.join(output_folder1, slice_name1)
                            img.save(output_path)
                            print(f"Die Slice {exec_file} wurde als {slice_name1} gespeichert.")
                            index1 += 1

                        if ds.SeriesDescription == sequence_name2:
                            # Extrahiere das DICOM-Bild als Numpy-Array
                            pixel_array2 = ds.pixel_array.astype('float32')
                            pixel_array2 /= np.max(pixel_array2)

                            img2 = Image.fromarray(np.uint8(pixel_array2*255))

                            # Speichere das DICOM-Bild als PNG-Datei im Ausgabeverzeichnis
                            index_str2 = "{:05d}".format(index2)
                            slice_name2 = f"{index_str2}.png"
                            output_path2 = os.path.join(output_folder2, slice_name2)
                            img2.save(output_path2)
                            print(f"Die Slice {exec_file} wurde als {slice_name2} gespeichert.")
                            index2 += 1
                    else:
                        continue

    print("Vorgang abgeschlossen.")


# Beispielaufruf der Funktion
input_folder = '/work/yv312705/NYU_data/knee_mri_clinical_seq'
output_folder1 = '/home/yv312705/Code/diffusion_autoenc/FastMri/COR_pd'
output_folder2 = '/home/yv312705/Code/diffusion_autoenc/FastMri/COR_pd_fs'
sequence_name1 = 'COR PD'
sequence_name2 = 'COR PD FS'

convert_dicom_to_png(input_folder, output_folder1, output_folder2, sequence_name1, sequence_name2)