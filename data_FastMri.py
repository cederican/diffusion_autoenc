import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2
from tqdm import tqdm
from PIL import Image, ImageEnhance



def convert_dicom_to_png(input_folder, output_folder1, output_folder2, output_folder3 ,output_folder4 ,output_folder5 ,sequence_name1, sequence_name2, sequence_name3, sequence_name4, sequence_name5):

    index1 = 00000
    index2 = 00000
    index3 = 00000
    index4 = 00000
    index5 = 00000

    start_index = 3001
    end_index = 3100

    index_stack_start = 13
    index_stack_end = 21
    


    subfolders = [f for f in sorted(os.listdir(input_folder)) if os.path.isdir(os.path.join(input_folder, f))]

    for folder_name in tqdm(subfolders[start_index:end_index], total=end_index-start_index):
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
                for exec_file in tqdm(exec_files[index_stack_start:index_stack_end], total=index_stack_end-index_stack_start):
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
                            #print(f"Die Slice {exec_file} wurde als {slice_name1} gespeichert.")
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
                            #print(f"Die Slice {exec_file} wurde als {slice_name2} gespeichert.")
                            index2 += 1

                        if ds.SeriesDescription == sequence_name3:
                            # Extrahiere das DICOM-Bild als Numpy-Array
                            pixel_array3 = ds.pixel_array.astype('float32')
                            pixel_array3 /= np.max(pixel_array3)

                            img3 = Image.fromarray(np.uint8(pixel_array3*255))

                            # Speichere das DICOM-Bild als PNG-Datei im Ausgabeverzeichnis
                            index_str3 = "{:05d}".format(index1)
                            slice_name3 = f"{index_str3}.png"
                            output_path3 = os.path.join(output_folder3, slice_name3)
                            img3.save(output_path3)
                            #print(f"Die Slice {exec_file} wurde als {slice_name2} gespeichert.")
                            index1 += 1

                        if ds.SeriesDescription == sequence_name4:
                            # Extrahiere das DICOM-Bild als Numpy-Array
                            pixel_array4 = ds.pixel_array.astype('float32')
                            pixel_array4 /= np.max(pixel_array4)

                            img4 = Image.fromarray(np.uint8(pixel_array4*255))

                            # Speichere das DICOM-Bild als PNG-Datei im Ausgabeverzeichnis
                            index_str4 = "{:05d}".format(index4)
                            slice_name4 = f"{index_str4}.png"
                            output_path4 = os.path.join(output_folder4, slice_name4)
                            img4.save(output_path4)
                            #print(f"Die Slice {exec_file} wurde als {slice_name2} gespeichert.")
                            index4 += 1

                        if ds.SeriesDescription == sequence_name5:
                            # Extrahiere das DICOM-Bild als Numpy-Array
                            pixel_array5 = ds.pixel_array.astype('float32')
                            pixel_array5 /= np.max(pixel_array5)

                            img5 = Image.fromarray(np.uint8(pixel_array5*255))

                            # Speichere das DICOM-Bild als PNG-Datei im Ausgabeverzeichnis
                            index_str5 = "{:05d}".format(index4)
                            slice_name5 = f"{index_str5}.png"
                            output_path5 = os.path.join(output_folder5, slice_name5)
                            img5.save(output_path5)
                            #print(f"Die Slice {exec_file} wurde als {slice_name2} gespeichert.")
                            index4 += 1
                    else:
                        continue

    print("Vorgang abgeschlossen.")


# Beispielaufruf der Funktion
input_folder = '/work/yv312705/NYU_data/knee_mri_clinical_seq'
output_folder1 = '/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier/a_cor_pd'
output_folder2 = '/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier/b_cor_pd_fs'
output_folder3 = '/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier_contr/a_pd'
output_folder4 = '/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier_contr/d_t2_fs'
output_folder5 = '/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier_contr/d_t2_fs'
sequence_name1 = 'COR PD'
sequence_name2 = 'COR PD FS'
sequence_name3 = 'wSAG PD'
sequence_name4 = 'wSAG T2 FS'
sequence_name5 = 'wAX T2 FS'

convert_dicom_to_png(input_folder, output_folder1, output_folder2, output_folder3 ,output_folder4 ,output_folder5 ,sequence_name1, sequence_name2, sequence_name3, sequence_name4, sequence_name5)