from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from config import *
from dataset import *
import pandas as pd
import json
import os
import copy

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
import torch
from torch.utils.data.dataset import ConcatDataset, TensorDataset



class ZipLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __len__(self):
        return len(self.loaders[0])

    def __iter__(self):
        for each in zip(*self.loaders):
            yield each


class ClsModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode.is_manipulate()
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf

        # preparations
        if conf.train_mode == TrainMode.manipulate:
            # this is only important for training!
            # the latent is freshly inferred to make sure it matches the image
            # manipulating latents require the base model
            self.model = conf.make_model_conf().make_model()
            self.ema_model = copy.deepcopy(self.model)
            self.model.requires_grad_(False)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()

            if conf.pretrain is not None:
                print(f'loading pretrain ... {conf.pretrain.name}')
                state = torch.load(conf.pretrain.path, map_location='cpu')
                print('step:', state['global_step'])
                self.load_state_dict(state['state_dict'], strict=False)

            # load the latent stats
            if conf.manipulate_znormalize:
                print('loading latent stats ...')
                state = torch.load(conf.latent_infer_path)
                self.conds = state['conds']
                self.register_buffer('conds_mean',
                                     state['conds_mean'][None, :])
                self.register_buffer('conds_std', state['conds_std'][None, :])
            else:
                self.conds_mean = None
                self.conds_std = None

        if conf.manipulate_mode in [ManipulateMode.celebahq_all]:
            num_cls = len(CelebAttrDataset.id_to_cls)
        elif conf.manipulate_mode.is_single_class():
            num_cls = 1
        elif conf.manipulate_mode.is_mri():
            num_cls = len(MriAttrDataset.id_to_cls)
        elif conf.manipulate_mode.is_mri_cls_eval():
            num_cls = len(MriAttrDataset.id_to_cls)
        else:
            raise NotImplementedError()

        # classifier
        if conf.train_mode == TrainMode.manipulate:
            # latent manipluation requires only a linear classifier
            self.classifier = nn.Linear(conf.style_ch, num_cls)
        else:
            raise NotImplementedError()

        self.ema_classifier = copy.deepcopy(self.classifier)

    def state_dict(self, *args, **kwargs):
        # don't save the base model
        out = {}
        for k, v in super().state_dict(*args, **kwargs).items():
            if k.startswith('model.'):
                pass
            elif k.startswith('ema_model.'):
                pass
            else:
                out[k] = v
        return out

    def load_state_dict(self, state_dict, strict: bool = None):
        if self.conf.train_mode == TrainMode.manipulate:
            # change the default strict => False
            if strict is None:
                strict = False
        else:
            if strict is None:
                strict = True
        return super().load_state_dict(state_dict, strict=strict)

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def load_dataset(self):
        if self.conf.manipulate_mode == ManipulateMode.d2c_fewshot:
            return CelebD2CAttrFewshotDataset(
                cls_name=self.conf.manipulate_cls,
                K=self.conf.manipulate_shots,
                img_folder=data_paths['celeba'],
                img_size=self.conf.img_size,
                seed=self.conf.manipulate_seed,
                all_neg=False,
                do_augment=True,
            )
        elif self.conf.manipulate_mode == ManipulateMode.d2c_fewshot_allneg:
            # positive-unlabeled classifier needs to keep the class ratio 1:1
            # we use two dataloaders, one for each class, to stabiliize the training
            img_folder = data_paths['celeba']

            return [
                CelebD2CAttrFewshotDataset(
                    cls_name=self.conf.manipulate_cls,
                    K=self.conf.manipulate_shots,
                    img_folder=img_folder,
                    img_size=self.conf.img_size,
                    only_cls_name=self.conf.manipulate_cls,
                    only_cls_value=1,
                    seed=self.conf.manipulate_seed,
                    all_neg=True,
                    do_augment=True),
                CelebD2CAttrFewshotDataset(
                    cls_name=self.conf.manipulate_cls,
                    K=self.conf.manipulate_shots,
                    img_folder=img_folder,
                    img_size=self.conf.img_size,
                    only_cls_name=self.conf.manipulate_cls,
                    only_cls_value=-1,
                    seed=self.conf.manipulate_seed,
                    all_neg=True,
                    do_augment=True),
            ]
        elif self.conf.manipulate_mode == ManipulateMode.celebahq_all:
            return CelebHQAttrDataset(data_paths['celebahq'],
                                      self.conf.img_size,
                                      data_paths['celebahq_anno'],
                                      do_augment=True)
        
        # load mri dataset ##############################################################################

        elif self.conf.manipulate_mode == ManipulateMode.mri:
            return MriAttrDataset(data_paths['mrilmdb'],
                                  self.conf.img_size,
                                  data_paths['mri_anno'],
                                  do_augment=True)

        elif self.conf.manipulate_mode == ManipulateMode.mri_cls_eval:
            return MriAttrDataset(data_paths['mrilmdb_cls_eval'],
                                  self.conf.img_size,
                                  data_paths['mri_anno_cls_eval'],
                                  do_augment=False)
        

        else:
            raise NotImplementedError()

    def setup(self, stage=None) -> None:
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.load_dataset()
        if self.conf.manipulate_mode.is_fewshot():
            # repeat the dataset to be larger (speed up the training)
            if isinstance(self.train_data, list):
                # fewshot-allneg has two datasets
                # we resize them to be of equal sizes
                a, b = self.train_data
                self.train_data = [
                    Repeat(a, max(len(a), len(b))),
                    Repeat(b, max(len(a), len(b))),
                ]
            else:
                self.train_data = Repeat(self.train_data, 100_000)

    def train_dataloader(self):
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        if isinstance(self.train_data, list):
            dataloader = []
            for each in self.train_data:
                dataloader.append(
                    conf.make_loader(each, shuffle=True, drop_last=True))
            dataloader = ZipLoader(dataloader)
        else:
            dataloader = conf.make_loader(self.train_data,
                                          shuffle=True,
                                          drop_last=True)
        return dataloader
    
    def test_dataloader(self):
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        if isinstance(self.train_data, list):
            dataloader = []
            for each in self.train_data:
                dataloader.append(
                    conf.make_loader(each, shuffle=True, drop_last=True))
            dataloader = ZipLoader(dataloader)
        else:
            dataloader = conf.make_loader(self.train_data,
                                          shuffle=False,
                                          drop_last=True)
        return dataloader


    @property
    def batch_size(self):
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws
    
    ################################# integrate the edge loss
    def edge_loss(self, cond, pred):

        cond_gradients = torch.autograd.grad(outputs=pred_softmax[:,1], inputs=cond, grad_outputs=torch.ones_like(pred_softmax[:,1]), create_graph=True, retain_graph=True, only_inputs=True)[0]
        edge_loss = torch.mean(torch.abs(cond_gradients))
        return edge_loss
    

    def training_step(self, batch, batch_idx):
        self.ema_model: BeatGANsAutoencModel
        if isinstance(batch, tuple):
            a, b = batch
            imgs = torch.cat([a['img'], b['img']])
            labels = torch.cat([a['labels'], b['labels']])
        else:
            imgs = batch['img']
            # print(f'({self.global_rank}) imgs:', imgs.shape)
            labels = batch['labels']

        if self.conf.train_mode == TrainMode.manipulate:
            self.ema_model.eval()
            with torch.no_grad():
                # (n, c)
                cond = self.ema_model.encoder(imgs)

            cond.requires_grad_(True) 
            if self.conf.manipulate_znormalize:
                cond = self.normalize(cond)

            # (n, cls)
            pred = self.classifier.forward(cond)
            pred_ema = self.ema_classifier.forward(cond)
        elif self.conf.train_mode == TrainMode.manipulate_img:
            # (n, cls)
            pred = self.classifier.forward(imgs)
            pred_ema = None
        elif self.conf.train_mode == TrainMode.manipulate_imgt:
            t, weight = self.T_sampler.sample(len(imgs), imgs.device)
            imgs_t = self.sampler.q_sample(imgs, t)
            pred = self.classifier.forward(imgs_t, t=t)
            pred_ema = None
            print('pred:', pred.shape)
        else:
            raise NotImplementedError()

        if self.conf.manipulate_mode.is_celeba_attr():
            gt = torch.where(labels > 0,
                             torch.ones_like(labels).float(),
                             torch.zeros_like(labels).float())   
        elif self.conf.manipulate_mode.is_mri():
            gt = torch.where(labels > 0,
                             torch.ones_like(labels).float(),
                             torch.zeros_like(labels).float())
        elif self.conf.manipulate_mode == ManipulateMode.relighting:
            gt = labels
        else:
            raise NotImplementedError()

        if self.conf.manipulate_loss == ManipulateLossType.bce:
            loss = F.binary_cross_entropy_with_logits(pred, gt)

            #cond_gradients = torch.autograd.grad(outputs=pred, inputs=cond, grad_outputs=torch.ones_like(pred), create_graph=True, retain_graph=True, only_inputs=True)[0]
            #edge_loss = torch.mean(torch.abs(cond_gradients))

            #total_loss = loss + edge_loss* 10

            #diff = torch.abs(cond[:, 1:] - cond[:, :-1])

            #edge_loss = torch.mean(diff)

            #total_loss = loss + edge_loss

            if pred_ema is not None:
                loss_ema = F.binary_cross_entropy_with_logits(pred_ema, gt)

                #cond_gradients_ema = torch.autograd.grad(outputs=pred_ema, inputs=cond, grad_outputs=torch.ones_like(pred_ema), create_graph=True, retain_graph=True, only_inputs=True)[0]
                #edge_loss_ema = torch.mean(torch.abs(cond_gradients_ema))

                #total_loss_ema = loss_ema + edge_loss_ema

        elif self.conf.manipulate_loss == ManipulateLossType.mse:
            loss = F.mse_loss(pred, gt)
            if pred_ema is not None:
                loss_ema = F.mse_loss(pred_ema, gt)
        else:
            raise NotImplementedError()

        #self.scheduler.step()
        #print(self.scheduler.get_lr())

        self.log('loss', loss)
        #self.log('edge_loss', edge_loss)
        self.log('loss_ema', loss_ema)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int,
                           dataloader_idx: int) -> None:
        ema(self.classifier, self.ema_classifier, self.conf.ema_decay)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.classifier.parameters(),
                                 lr=self.conf.lr,
                                 weight_decay=self.conf.weight_decay)
        
        # scheduler implementation
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,
                            #gamma=self.conf.gamma)
        #self.scheduler = scheduler
        return [optim] #[scheduler]
    
    def test_step(self, batch, batch_idx):
        
        self.ema_model: BeatGANsAutoencModel
        if isinstance(batch, tuple):
            a, b = batch
            imgs = torch.cat([a['img'], b['img']])
            labels = torch.cat([a['labels'], b['labels']])
        else:
            imgs = batch['img']
            labels = batch['labels']

        if self.conf.train_mode == TrainMode.manipulate:
            self.ema_model.eval()
            with torch.no_grad():
                cond = self.ema_model.encoder(imgs)

            if self.conf.manipulate_znormalize:
                cond = self.normalize(cond)

            pred = self.classifier.forward(cond)

            _ , predict = torch.max(pred, dim=1)
            #predict = predict.item()
            print('Predict:', predict.tolist())
            label_index = [torch.nonzero(row > 0) for row in labels]
            print('Truth:', label_index)

            pred_ema = self.ema_classifier.forward(cond)
        elif self.conf.train_mode == TrainMode.manipulate_img:
            pred = self.classifier.forward(imgs)
            pred_ema = None
        elif self.conf.train_mode == TrainMode.manipulate_imgt:
            t, weight = self.T_sampler.sample(len(imgs), imgs.device)
            imgs_t = self.sampler.q_sample(imgs, t)
            pred = self.classifier.forward(imgs_t, t=t)
            pred_ema = None
            print('pred:', pred.shape)
        else:
            raise NotImplementedError()

        if self.conf.manipulate_mode.is_celeba_attr():
            gt = torch.where(labels > 0,
                             torch.ones_like(labels).float(),
                             torch.zeros_like(labels).float())
        elif self.conf.manipulate_mode.is_mri():
            gt = torch.where(labels > 0,
                             torch.ones_like(labels).float(),
                             torch.zeros_like(labels).float())
        elif self.conf.manipulate_mode.is_mri_cls_eval():
            gt = torch.where(labels > 0,
                             torch.ones_like(labels).float(),
                             torch.zeros_like(labels).float())
        elif self.conf.manipulate_mode == ManipulateMode.relighting:
            gt = labels
        else:
            raise NotImplementedError()

        if self.conf.manipulate_loss == ManipulateLossType.bce:
            loss = F.binary_cross_entropy_with_logits(pred, gt)
            print('Loss:', loss.item())
            if pred_ema is not None:
                loss_ema = F.binary_cross_entropy_with_logits(pred_ema, gt)
                print('Loss EMA:', loss_ema.item())
        elif self.conf.manipulate_loss == ManipulateLossType.mse:
            loss = F.mse_loss(pred, gt)
            if pred_ema is not None:
                loss_ema = F.mse_loss(pred_ema, gt)
        else:
            raise NotImplementedError()
        

        self.log('val_loss', loss)
        self.log('val_loss_ema', loss_ema)
        return loss


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


def train_cls(conf: TrainConfig, gpus, mode: str = 'train'):
    print('conf:', conf.name)
    model = ClsModel(conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(
        dirpath=f'{conf.logdir}',
        save_last=True,
        save_top_k=1,
        # every_n_train_steps=conf.save_every_samples //
        # conf.batch_size_effective,
    )
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    # from pytorch_lightning.

    plugins = []
    if len(gpus) == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin
        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        resume_from_checkpoint=resume,
        gpus=gpus,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
        ],
        replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
        track_grad_norm=False
    )

    if mode == 'train':
        trainer.fit(model)
    elif mode == 'eval':
        #dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
                          #batch_size=conf.batch_size)
        setup = model.setup()
        testdataloader = model.test_dataloader()
        eval_path = conf.eval_path or checkpoint_path
        print('loading from:', eval_path)
        state = torch.load(eval_path, map_location='cpu')
        print('step:', state['global_step'])
        model.load_state_dict(state['state_dict'])

        out = trainer.test(model, dataloaders=testdataloader)
        out = out[0]
        print(out)

        #loss_values = []
        #for batch in testdataloader:
            #output = model.test_step(batch, batch_idx=0)  # `batch_idx` kann beliebig gewählt werden
            #loss_values.append(output.item())


        if get_rank() == 0:
            # save to tensorboard
            for k, v in out.items():
                tb_logger.experiment.add_scalar(
                    k, v, state['global_step'] * conf.batch_size_effective)
    else:
        raise NotImplementedError()         