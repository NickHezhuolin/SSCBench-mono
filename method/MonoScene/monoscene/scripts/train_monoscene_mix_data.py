# from monoscene.data.kitti_360.kitti_360_dm import Kitti360DataModule
from monoscene.data.mix_dataset.mix_dataset_dm import MixDataModule
from monoscene.data.kitti_360.params import (
    kitti_360_unified_class_frequencies,
    kitti_360_class_names,
)

from monoscene.data.nuscenes.params import (
    nuscenes_class_frequencies,
    nuscenes_class_names,
)

from monoscene.data.waymo.params import (
    waymo_class_frequencies,
    waymo_class_names,
)

from torch.utils.data.dataloader import DataLoader
from monoscene.models.monoscene import MonoScene
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch

hydra.output_subdir = None


@hydra.main(config_name="../config/monoscene_mix_data.yaml")
def main(config: DictConfig):
    exp_name = config.exp_prefix
    exp_name += "_{}_{}".format(config.dataset, config.run)
    exp_name += "_FrusSize_{}".format(config.frustum_size)
    exp_name += "_nRelations{}".format(config.n_relations)
    exp_name += "_WD{}_lr{}".format(config.weight_decay, config.lr)

    if config.CE_ssc_loss:
        exp_name += "_CEssc"
    if config.geo_scal_loss:
        exp_name += "_geoScalLoss"
    if config.sem_scal_loss:
        exp_name += "_semScalLoss"
    if config.fp_loss:
        exp_name += "_fpLoss"

    if config.relation_loss:
        exp_name += "_CERel"
    if config.context_prior:
        exp_name += "_3DCRP"

    # Setup dataloaders
    class_names = nuscenes_class_names
    max_epochs = 30
    logdir = config.kitti_360_logdir
    full_scene_size = (256, 256, 32)
    project_scale = 2
    feature = 64
    n_classes = len(nuscenes_class_names)
        
    # Mix class_frequencies
    mix_class_frequencies = kitti_360_unified_class_frequencies + waymo_class_frequencies
    
    class_weights = torch.from_numpy(
        1 / np.log(mix_class_frequencies + 0.001)
    )
    data_module = MixDataModule(
        kitti360_dataset_root=config.kitti_360_root,
        kitti360_dataset_preprocess_root=config.kitti_360_preprocess_root,
        nuscenes_dataset_root=config.nuscenes_root,
        nuscenes_dataset_preprocess_root=config.nuscenes_preprocess_root,
        waymo_dataset_root=config.waymo_root,
        waymo_dataset_preprocess_root=config.waymo_preprocess_root,
        frustum_size=config.frustum_size,
        project_scale=project_scale,
        batch_size=int(config.batch_size / config.n_gpus),
        num_workers=int(config.num_workers_per_gpu),
    )


    project_res = ["1"]
    if config.project_1_2:
        exp_name += "_Proj_2"
        project_res.append("2")
    if config.project_1_4:
        exp_name += "_4"
        project_res.append("4")
    if config.project_1_8:
        exp_name += "_8"
        project_res.append("8")

    print(exp_name)

    # Initialize MonoScene model
    model = MonoScene(
        dataset=config.dataset,
        frustum_size=config.frustum_size,
        project_scale=project_scale,
        n_relations=config.n_relations,
        fp_loss=config.fp_loss,
        feature=feature,
        full_scene_size=full_scene_size,
        project_res=project_res,
        n_classes=n_classes,
        class_names=class_names,
        context_prior=config.context_prior,
        relation_loss=config.relation_loss,
        CE_ssc_loss=config.CE_ssc_loss,
        sem_scal_loss=config.sem_scal_loss,
        geo_scal_loss=config.geo_scal_loss,
        lr=config.lr,
        weight_decay=config.weight_decay,
        class_weights=class_weights,
    )

    if config.enable_log:
        logger = TensorBoardLogger(save_dir=logdir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val/mIoU",
                save_top_k=1,
                mode="max",
                filename="{epoch:03d}-{val/mIoU:.5f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    if os.path.isfile(model_path):
        # Continue training from last.ckpt
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
        )
    else:
        # Train from scratch
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
        )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()