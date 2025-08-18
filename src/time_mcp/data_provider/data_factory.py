import lightning as L
from time_mcp.data_provider.data_loader import Dataset_Custom

from torch.utils.data import DataLoader
from minio import Minio
class LightLoader(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        minio_client: Minio,
        data_name: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        features: str = "S",
        target: str = "OT",
        timeenc=0,
        freq="t",
        train_only=False,
        num_workers=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.client = minio_client
        self.data_name = data_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = Dataset_Custom(
                minio_client=self.client,
                data_name=self.data_name,
                flag="train",
                size=[self.seq_len, self.label_len, self.pred_len],
                features=self.features,
                target=self.target,
                timeenc=self.timeenc,
                freq=self.freq,
                train_only=self.train_only,
            )
            self.vali_set = Dataset_Custom(
                minio_client=self.client,
                data_name=self.data_name,
                flag="val",
                size=[self.seq_len, self.label_len, self.pred_len],
                features=self.features,
                target=self.target,
                timeenc=self.timeenc,
                freq=self.freq,
                train_only=self.train_only,
            )
        if stage == "test":
            self.test_set = Dataset_Custom(
                minio_client=self.client,
                data_name=self.data_name,
                flag="test",
                size=[self.seq_len, self.label_len, self.pred_len],
                features=self.features,
                target=self.target,
                timeenc=self.timeenc,
                freq=self.freq,
                train_only=self.train_only,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.vali_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

