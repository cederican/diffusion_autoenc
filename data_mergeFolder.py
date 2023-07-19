import os
from shutil import copyfile
from PIL import Image

# --------------------- helper function to merge folder into one big folder for a trainingdataset -------------------------------


def merge_png_folders(folder1, folder2, folder3, target_folder1, target_folder2, target_folder3):
    """
    Merges two folders containing PNG files into one folder,
    with the file names numbered in sequence starting from 00000.png.
    """
    #if not os.path.exists(target_folder):
    #   os.makedirs(target_folder)

    files1 = [f for f in sorted(os.listdir(folder1)) if f.endswith('.png')][:5000]

    files2 = [f for f in sorted(os.listdir(folder2)) if f.endswith('.png')][15000:20000]

    files3 = [f for f in sorted(os.listdir(folder3)) if f.endswith('.png')][30200:35200]
    
    #files4 = [f for f in sorted(os.listdir(folder4)) if f.endswith('.png')]

    #files5 = [f for f in sorted(os.listdir(folder5)) if f.endswith('.png')]

    #files6 = [f for f in sorted(os.listdir(folder6)) if f.endswith('.png')]

    
    for i, file1 in enumerate(files1):
        source_file = os.path.join(folder1, file1)
        image = Image.open(source_file)
        resized_image = image.resize((128,128))
        target_file = os.path.join(target_folder1, f"{i:05}.png")
        resized_image.save(target_file)
        #copyfile(source_file, target_file)
    
    for i, file2 in enumerate(files2):
        source_file = os.path.join(folder2, file2)
        image = Image.open(source_file)
        resized_image = image.resize((128,128))
        target_file = os.path.join(target_folder2, f"{i:05}.png")
        resized_image.save(target_file)
        #copyfile(source_file, target_file)

    for i, file3 in enumerate(files3):
        source_file = os.path.join(folder3, file3)
        image = Image.open(source_file)
        resized_image = image.resize((128,128))
        target_file = os.path.join(target_folder3, f"{i:05}.png")
        resized_image.save(target_file)
        #copyfile(source_file, target_file)
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
folder1 = '/home/yv312705/Code/diffusion_autoenc/datasets/FastMRI_cor2.png'
folder2 = '/home/yv312705/Code/diffusion_autoenc/datasets/FastMRI_cor2.png'
folder3 = '/home/yv312705/Code/diffusion_autoenc/datasets/FastMRI_cor2.png'
#folder4 = '/home/yv312705/Code/diffusion_autoenc/FastMri/AX_t2_fs'
#folder5 = '/home/yv312705/Code/diffusion_autoenc/FastMri/SAG_pd'
#folder6 = '/home/yv312705/Code/diffusion_autoenc/FastMri/SAG_t2_fs'
target_folder1 = '/home/yv312705/Code/diffusion_autoenc/eval_metrics/realPD/PD'
target_folder2 = '/home/yv312705/Code/diffusion_autoenc/eval_metrics/realPDfs/PD fs'
target_folder3 = '/home/yv312705/Code/diffusion_autoenc/eval_metrics/realT1/T1'
#merge_png_folders(folder1, folder2, folder3, folder4, folder5, folder6, target_folder)
merge_png_folders(folder1, folder2, folder3, target_folder1, target_folder2, target_folder3)