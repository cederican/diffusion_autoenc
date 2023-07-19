import numpy as np
import os
import cv2
from tqdm import tqdm

start_index = 1151
end_index = 1250

def save_slices_as_png(input_dir, output_dir):
    # Get a list of all .npy files in the input directory
    file_list = []
    index = 00000
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for orientation_type in ["axial", "coronal", "sagittal"]:
            if not orientation_type == "coronal":
                continue
            orientation_type_path = os.path.join(folder_path, orientation_type)
            if not os.path.isdir(orientation_type_path):
                continue
            for file_name in sorted(os.listdir(orientation_type_path)):
                if file_name.endswith(".npy"):
                    file_list.append(os.path.join(orientation_type_path, file_name))

    # Iterate through the list of files and save each slice as a separate .png file
    for file_path in tqdm(file_list[start_index:end_index], total=end_index-start_index):  # loop through only the first 10 files
        # Load the .npy file
        image_data = np.load(file_path)
        

        # Get the metadata
        file_name = os.path.basename(file_path)
        name , ext = file_name.split(".")
        path_split = os.path.split(file_path)[0].split(os.sep)
        orientation_name = path_split[-1]

        # Loop through the slices and save each one as a separate .png file
        for i in range(image_data.shape[0]-28):
            index_str = "{:05d}".format(index)
            #slice_name = f"{orientation_name}_{name}_{index_str}.png"
            slice_name = f"{index_str}.png"
            slice_path = os.path.join(output_dir, slice_name)
            #AE benötigt 3 channel, dupliziere gray in 3 channel
            img_3ch = np.stack([image_data[i+14]] * 3, axis=2)
            cv2.imwrite(slice_path, img_3ch)
            index += 1

data_dir = "/home/yv312705/Code/diffusion_autoenc/datasets/MRNet-v1.0"
out_dir = "/home/yv312705/Code/diffusion_autoenc/FastMri/test_classifier/c_cor_t1"
save_slices_as_png(data_dir, out_dir)

