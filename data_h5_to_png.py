import h5py
import numpy as np
import matplotlib.pyplot as plt

def h5_to_png(h5_file_path, output_folder):
    # Öffne die .h5-Datei im Lesemodus
    with h5py.File(h5_file_path, "r") as f:

        print(list(f.keys()))
        
        # Lies die Daten aus dem Dataset 'slices' ein
        slices = f["ismrmrd_header"][:]
        # Gehe durch alle Slices und speichere sie als .png-Datei im Ausgabeordner
        for i, slice in enumerate(slices):
            output_path = f"{output_folder}/slice_{i:03}.png"
            plt.imsave(output_path, slice, cmap="gray")


h5_file_path = '/home/yv312705/Code/diffusion_autoenc/testdataset/file1000022.h5'
output_folder = '/home/yv312705/Code/diffusion_autoenc/testdataset'

h5_to_png(h5_file_path, output_folder)