from torch.utils.data.dataloader import DataLoader
from monoscene.data.kitti_360.kitti_360_dataset import Kitti360Dataset
from monoscene.data.nuscenes.nuscenes_dataset import NuScenesDataset
from monoscene.data.waymo.waymo_dataset import WaymoDataset
import pytorch_lightning as pl
from monoscene.data.kitti_360.collate import collate_fn
from monoscene.data.utils.torch_util import worker_init_fn
from torch.utils.data import ConcatDataset
from monoscene.data.mix_dataset.collate_mix_data import collate_fn_mix

class MixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        kitti360_dataset_root,
        kitti360_dataset_preprocess_root,
        nuscenes_dataset_root,
        nuscenes_dataset_preprocess_root,
        waymo_dataset_root,
        waymo_dataset_preprocess_root,
        project_scale=2,
        frustum_size=4,
        batch_size=4, 
        num_workers=6,
        dataset="kitti_360_waymo",
    ):

        super().__init__()
        self.kitti360_dataset_root = kitti360_dataset_root
        self.kitti360_dataset_preprocess_root = kitti360_dataset_preprocess_root
        self.nuscenes_dataset_root = nuscenes_dataset_root
        self.nuscenes_dataset_preprocess_root = nuscenes_dataset_preprocess_root
        self.waymo_dataset_root = waymo_dataset_root
        self.waymo_dataset_preprocess_root = waymo_dataset_preprocess_root
        self.project_scale = project_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        self.dataset=dataset

    def setup(self, stage=None):
        kitti360_train_ds = Kitti360Dataset(
            split="train",
            root=self.kitti360_dataset_root,
            preprocess_root=self.kitti360_dataset_preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
        )
        
        nuscenes_train_ds = NuScenesDataset(
            split="train",
            root=self.nuscenes_dataset_root,
            preprocess_root=self.nuscenes_dataset_preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
        )
        
        waymo_train_ds = WaymoDataset(
            split="train",
            root=self.waymo_dataset_root,
            preprocess_root=self.waymo_dataset_preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
        )
        
        # double dataset training
        if self.dataset == "kitti_360_nuscenes":
            self.train_ds = ConcatDataset([kitti360_train_ds, nuscenes_train_ds])
            print("Using kitti_360 and nuscenes datasets for training.")
        elif self.dataset == "kitti_360_waymo":
            self.train_ds = ConcatDataset([kitti360_train_ds, waymo_train_ds])
            print("Using kitti_360 and waymo datasets for training.")
        elif self.dataset == "nuscenes_waymo":
            self.train_ds = ConcatDataset([nuscenes_train_ds, waymo_train_ds])
            print("Using nuscenes and waymo datasets for training.")
        elif self.dataset == "kitti_360_nuscenes_waymo":
            # triple dataset training
            self.train_ds = ConcatDataset([kitti360_train_ds, nuscenes_train_ds, waymo_train_ds])
            print("Using kitti_360, nuscenes, and waymo datasets for training.")
        else:
            raise ValueError(f"Unknown dataset combination: {self.dataset}.")

        self.val_ds = Kitti360Dataset(
            split="val",
            root=self.kitti360_dataset_root,
            preprocess_root=self.kitti360_dataset_preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
        ) 
        # NuScenesDataset(
        #     split="val",
        #     root=self.nuscenes_dataset_root,
        #     preprocess_root=self.nuscenes_dataset_preprocess_root,
        #     project_scale=self.project_scale,
        #     frustum_size=self.frustum_size,
        #     fliplr=0,
        #     color_jitter=None,
        # )

        self.test_ds = Kitti360Dataset(
            split="test",
            root=self.kitti360_dataset_root,
            preprocess_root=self.kitti360_dataset_preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
        )
        # NuScenesDataset(
        #     split="test",
        #     root=self.nuscenes_dataset_root,
        #     preprocess_root=self.nuscenes_dataset_preprocess_root,
        #     project_scale=self.project_scale,
        #     frustum_size=self.frustum_size,
        #     fliplr=0,
        #     color_jitter=None,
        # )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_mix,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
