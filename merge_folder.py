import os
from shutil import copyfile

def merge_png_folders(folder1, folder2, target_folder):
    """
    Merges two folders containing PNG files into one folder,
    with the file names numbered in sequence starting from 000001.png.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all PNG files in the first folder
    files1 = [f for f in sorted(os.listdir(folder1)) if f.endswith('.png')][:12000]

    # List all PNG files in the second folder
    files2 = [f for f in sorted(os.listdir(folder2)) if f.endswith('.png')]

    # Copy files from folder1 to target folder with numbered file names
    for i, file1 in enumerate(files1):
        source_file = os.path.join(folder1, file1)
        target_file = os.path.join(target_folder, f"{i:05}.png")
        copyfile(source_file, target_file)

    # Copy files from folder2 to target folder with numbered file names
    for i, file2 in enumerate(files2):
        source_file = os.path.join(folder2, file2)
        target_file = os.path.join(target_folder, f"{i+len(files1)+1:05}.png")
        copyfile(source_file, target_file)


folder1 = '/home/yv312705/Code/diffusion_autoenc/datasets/MRNet_png_axial'
folder2 = '/home/yv312705/Code/diffusion_autoenc/datasets/Unet_darius_scaled'
target_folder = '/home/yv312705/Code/diffusion_autoenc/datasets/test_twow_ax'
merge_png_folders(folder1, folder2, target_folder)