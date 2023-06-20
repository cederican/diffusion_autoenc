import os
from shutil import copyfile

def merge_png_folders(folder1, folder2, folder3, target_folder):
    """
    Merges two folders containing PNG files into one folder,
    with the file names numbered in sequence starting from 00000.png.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all PNG files in the first folder
    files1 = [f for f in sorted(os.listdir(folder1)) if f.endswith('.png')]

    # List all PNG files in the second folder
    files2 = [f for f in sorted(os.listdir(folder2)) if f.endswith('.png')]

    files3 = [f for f in sorted(os.listdir(folder3)) if f.endswith('.png')]
    
    #files4 = [f for f in sorted(os.listdir(folder4)) if f.endswith('.png')]

    #files5 = [f for f in sorted(os.listdir(folder5)) if f.endswith('.png')]

    #files6 = [f for f in sorted(os.listdir(folder6)) if f.endswith('.png')]

    # Copy files from folder1 to target folder with numbered file names
    for i, file1 in enumerate(files1):
        source_file = os.path.join(folder1, file1)
        target_file = os.path.join(target_folder, f"{i:05}.png")
        copyfile(source_file, target_file)

    # Copy files from folder2 to target folder with numbered file names
    for i, file2 in enumerate(files2):
        source_file = os.path.join(folder2, file2)
        target_file = os.path.join(target_folder, f"{i+len(files1):05}.png")
        copyfile(source_file, target_file)

    for i, file3 in enumerate(files3):
        source_file = os.path.join(folder3, file3)
        target_file = os.path.join(target_folder, f"{i+len(files1)+len(files2):05}.png")
        copyfile(source_file, target_file)
    '''
    for i, file4 in enumerate(files4):
        source_file = os.path.join(folder4, file4)
        target_file = os.path.join(target_folder, f"{i+len(files1)+len(files2)+len(files3):05}.png")
        copyfile(source_file, target_file)
    
    for i, file5 in enumerate(files5):
        source_file = os.path.join(folder5, file5)
        target_file = os.path.join(target_folder, f"{i+len(files1)+len(files2)+len(files3)+len(files4):05}.png")
        copyfile(source_file, target_file)

    for i, file6 in enumerate(files6):
        source_file = os.path.join(folder6, file6)
        target_file = os.path.join(target_folder, f"{i+len(files1)+len(files2)+len(files3)+len(files4)+len(files5):05}.png")
        copyfile(source_file, target_file)

    '''
folder1 = '/home/yv312705/Code/diffusion_autoenc/FastMri/COR_pd'
folder2 = '/home/yv312705/Code/diffusion_autoenc/FastMri/COR_pd_fs'
folder3 = '/home/yv312705/Code/diffusion_autoenc/MRNet/COR_t1'
#folder4 = '/home/yv312705/Code/diffusion_autoenc/FastMri/AX_t2_fs'
#folder5 = '/home/yv312705/Code/diffusion_autoenc/FastMri/SAG_pd'
#folder6 = '/home/yv312705/Code/diffusion_autoenc/FastMri/SAG_t2_fs'
target_folder = '/home/yv312705/Code/diffusion_autoenc/datasets/FastMRI_cor2.png'
#merge_png_folders(folder1, folder2, folder3, folder4, folder5, folder6, target_folder)
merge_png_folders(folder1, folder2, folder3, target_folder)