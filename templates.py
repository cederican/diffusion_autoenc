from experiment import *


def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf

'''
##############################################################################################################
'''
'''
# ----------------- configuration function for 128x128 MRI scans -----------------------
'''

def mri_autoenc():
    conf = autoenc_base()
    conf.data_name = 'mrilmdb'
    conf.img_size = 128                                    #edit for 128 or 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 3, 4)                     #edit for 128 or 256
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)         #edit for 128 or 256
    conf.eval_ema_every_samples = 2_000_000                      # edit for dataset
    conf.eval_every_samples = 2_000_000
    conf.eval_num_images = 10_000                              # how many images for FID Score
    conf.total_samples = 20_000_000                             # edit for dataset
    conf.batch_size = 16
    conf.scale_up_gpus(2)
    conf.make_model_conf()
    conf.name = 'mri_autoenc_nine'
    return conf

'''
# ---------------- pretrain function, if checkpoints are available
'''
def pretrain_mri():
    conf = mri_autoenc()
    conf.pretrain = PretrainConfig(
        name='10M',
        path=f'checkpoints/{mri_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{mri_autoenc().name}/latent.pkl'
    return conf

'''
# -------------------- ddpm config function
'''
def mri_ddpm():
    conf = ddpm()
    conf.data_name = 'mrilmdb'
    conf.warmup = 0
    conf.total_samples = 4_000_000                   # edit for dataset
    conf.img_size = 128                         #edit for minor
    conf.net_ch = 128
    # channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)  
    conf.eval_every_samples = 200_000                # edit for dataset
    conf.eval_ema_every_samples = 200_000            # edit for dataset
    conf.eval_num_images = 200 
    conf.batch_size = 16
    conf.scale_up_gpus(2)
    conf.make_model_conf()
    conf.name = 'mri_ddpm'
    return conf

'''
#################################################################################################################
'''

def ffhq128_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 128
    conf.net_ch = 128
    # channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq128_autoenc_base():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.scale_up_gpus(4)
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq256_autoenc():
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc'
    return conf


def ffhq256_autoenc_eco():
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc_eco'
    return conf


def ffhq128_ddpm_72M():
    conf = ffhq128_ddpm()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_ddpm_72M'
    return conf


def ffhq128_autoenc_72M():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_autoenc_72M'
    return conf


def ffhq128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_ddpm_130M'
    return conf


def ffhq128_autoenc_130M():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_autoenc_130M'
    return conf


def pretrain_celeba64d2c_72M():
    conf = celeba64d2c_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{celeba64d2c_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{celeba64d2c_autoenc().name}/latent.pkl'
    return conf


def pretrain_ffhq128_autoenc72M():
    conf = ffhq128_autoenc_base()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{ffhq128_autoenc_72M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_72M().name}/latent.pkl'
    return conf


def pretrain_ffhq128_autoenc130M():
    conf = ffhq128_autoenc_base()
    conf.pretrain = PretrainConfig(
        name='130M',
        path=f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
    return conf


def pretrain_ffhq256_autoenc():
    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'
    return conf


def pretrain_horse128():
    conf = horse128_autoenc()
    conf.pretrain = PretrainConfig(
        name='82M',
        path=f'checkpoints/{horse128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{horse128_autoenc().name}/latent.pkl'
    return conf


def pretrain_bedroom128():
    conf = bedroom128_autoenc()
    conf.pretrain = PretrainConfig(
        name='120M',
        path=f'checkpoints/{bedroom128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{bedroom128_autoenc().name}/latent.pkl'
    return conf
