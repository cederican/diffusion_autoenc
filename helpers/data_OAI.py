import os
import subprocess
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def convert_dicom_to_png(dicom_folder, output_folder, sequence_name):
    # Erstelle das Ausgabeverzeichnis, falls es nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Liste alle Ordner im angegebenen DICOM-Verzeichnis auf
    dicom_folders = [f for f in sorted(os.listdir(dicom_folder)) if os.path.isdir(os.path.join(dicom_folder, f))]

    # Durchlaufe jeden Ordner
    for folder_name in dicom_folders:
        folder_path = os.path.join(dicom_folder, folder_name)

        # Liste alle Dateien im aktuellen Ordner auf
        files = sorted(os.listdir(folder_path))

        # Suche nach den Exec-Dateien im aktuellen Ordner
        exec_files = [f for f in files if not os.path.isdir(os.path.join(folder_path, f))]

        # Durchlaufe die Exec-Dateien
        for exec_file in exec_files:
            exec_file_path = os.path.join(folder_path, exec_file)

            ds = pydicom.dcmread(exec_file_path)

            # Führe die Exec-Datei aus, um die DICOM-Bilder zu generieren
            subprocess.run([exec_file_path])

            # Liste alle DICOM-Dateien im aktuellen Ordner auf
            dicom_files = [f for f in files if f.endswith('.dcm')]

            # Durchlaufe die DICOM-Dateien
            for dicom_file in dicom_files:
                dicom_file_path = os.path.join(folder_path, dicom_file)

                # Lese die DICOM-Datei
                ds = pydicom.dcmread(dicom_file_path)

                # Überprüfe, ob die aktuelle DICOM-Datei die gewünschte Bildsequenz enthält
                if ds.SeriesDescription == sequence_name:
                    # Extrahiere das DICOM-Bild als Numpy-Array
                    img = ds.pixel_array

                    # Speichere das DICOM-Bild als PNG-Datei im Ausgabeverzeichnis
                    output_filename = os.path.splitext(dicom_file)[0] + '.png'
                    output_path = os.path.join(output_folder, output_filename)
                    plt.imsave(output_path, img, cmap='gray')
                    print(f"Die Slice {dicom_file} wurde als {output_filename} gespeichert.")

# Beispielaufruf der Funktion
dicom_folder = '/home/yv312705/Code/diffusion_autoenc/testforoai/20060323'
output_folder = '/home/yv312705/Code/diffusion_autoenc/testforoai/20060323/output'
sequence_name = 'Sag_IW_TSE_RIGHT'

convert_dicom_to_png(dicom_folder, output_folder, sequence_name)