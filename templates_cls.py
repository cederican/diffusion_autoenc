from templates import *




'''
###################################################################################################
classifier config function
'''
def mri_autoenc_cls():
    conf = mri_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.mri
    conf.manipulate_loss = ManipulateLossType.bce
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{mri_autoenc().name}/latent.pkl'
    conf.batch_size = 2            #edit for dataset
    conf.lr = 1e-5
    conf.total_samples = 300_000     #edit for dataset
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{mri_autoenc().name}/last.ckpt',
    )
    conf.name = 'mri_autoenc_cls_eight_justcontrast'
    return conf

def mri_autoenc_cls_eval():
    conf = mri_autoenc()
    conf.data_name = 'mrilmdb_cls_eval'
    conf.eval_path = '/home/yv312705/Code/diffusion_autoenc/checkpoints/mri_autoenc_cls_six/last.ckpt'
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.mri_cls_eval
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{mri_autoenc().name}/latent.pkl'
    conf.batch_size = 2            #edit for dataset
    conf.lr = 1e-3
    conf.total_samples = 300_000     #edit for dataset
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{mri_autoenc().name}/last.ckpt',
    )
    conf.name = 'mri_autoenc_cls_eval_six_edgeloss'
    return conf


def ffhq128_autoenc_cls():
    conf = ffhq128_autoenc_130M()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
    )
    conf.name = 'ffhq128_autoenc_cls'
    return conf


def ffhq256_autoenc_cls():
    '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
    conf = ffhq256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not FFHQ
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.name = 'ffhq256_autoenc_cls'
    return conf
