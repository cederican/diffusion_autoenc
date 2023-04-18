import numpy as np

def read_npy_file(filepath):
    """
    Reads a .npy file and prints out all the information it contains.
    """
    arr = np.load(filepath)
    print("Array shape:", arr.shape)
    print("Array dtype:", arr.dtype)
    print("Array size:", arr.size)
    print("Array ndim:", arr.ndim)
    print("Array data:", arr)


read_npy_file('/home/yv312705/Code/diffusion_autoenc/datasets/MRNet-v1.0/train/coronal/0000.npy')
