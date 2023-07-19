from templates import *
from templates_latent import *

# ------------- train for unconditioned sampling. not important in this thesis --------------------
if __name__ == '__main__':
    
    
    gpus = [0, 1]
    conf = mri_ddpm()
    train(conf, gpus=gpus)

    gpus = [0, 1]
    conf.eval_programs = ['fid10']
    train(conf, gpus=gpus, mode='eval')