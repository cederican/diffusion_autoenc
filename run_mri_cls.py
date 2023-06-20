from templates_cls import *
from experiment_classifier import *

if __name__ == '__main__':
    '''
    need to first train the diffae autoencoding model & infer the latents
    this requires only a single GPU.
    '''
    gpus = [0]
    conf = mri_autoenc_cls()
    train_cls(conf, gpus=gpus, mode='train')

    #gpus = [0]
    #conf = mri_autoenc_cls_eval()
    #train_cls(conf, gpus=gpus, mode='eval')
