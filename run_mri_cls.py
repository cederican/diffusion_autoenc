from templates_cls import *
from experiment_classifier import *

# -------------------------- script to train the classifier with pretrained diffae model and infered latents from run_mri.py ------------------
if __name__ == '__main__':
    '''
    need to first train the diffae autoencoding model & infer the latents
    this requires only a single GPU.
    '''
    gpus = [0]
    conf = mri_autoenc_cls()
    train_cls(conf, gpus=gpus, mode='train')

    # -------------- not import. evaluation of classifier is performed in test_mainpulate.py ------------------
    #gpus = [0]
    #conf = mri_autoenc_cls_eval()
    #train_cls(conf, gpus=gpus, mode='eval')
