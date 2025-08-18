import torch
import torch.nn as nn
import lightning as L
from fastmcp import Context
from time_mcp.utils import metrics
from time_mcp.utils.loss import get_loss_class
from time_mcp.utils.scheduler import get_scheduler
import asyncio

from time_mcp.data_provider.data_factory import LightLoader
import os
from minio import Minio

from concurrent.futures import ThreadPoolExecutor


async def n_linear_forecast(
    data_name,
    seq_len,
    pred_len,
    channels,
    lr,
    weight_decay,
    individual,
    max_epochs: int,
    batch_size: int,
    ctx: Context,
):
    """
    Asynchronous function to perform NLinear forecasting using PyTorch Lightning.
    Args:
        data_name (str): The name of the dataset to be used for forecasting.
        seq_len (int): The length of the input sequence.
        pred_len (int): The length of the prediction sequence.
        channels (int): The number of input channels in the data.
        lr (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay (L2 regularization) for the optimizer.
        scheduler_type (str): The type of learning rate scheduler to use. eg. "ExponentialLR", etc.
        loss_type (str): The type of loss function to use. eg. "SmoothL1Loss", "MSELoss", etc.
        individual (bool): Whether to use individual models for each target.
        max_epochs (int): The maximum number of training epochs.
        batch_size (int): The batch size for data loading.
        ctx (Context): The context object containing additional configuration or state.
    Returns:
        List[Dict[str, float]]: A list of dictionaries containing the test results for the model.
    Raises:
        ValueError: If any of the required parameters are invalid or missing.
        Exception: If an error occurs during model training or testing.
    """
    trainer = L.Trainer(max_epochs=max_epochs, enable_progress_bar=False)
    scheduler_type, sargs = await get_scheduler(ctx)
    loss = await get_loss_class(ctx)
    model = LightNLinear(
        seq_len,
        pred_len,
        channels,
        "forecast",
        lr=lr,
        weight_decay=weight_decay,
        scheduler=scheduler_type,
        loss=loss,
        individual=individual,
        **sargs,
    )

    minio_client = Minio(
        os.getenv("MINIO_ENDPOINT") or "default_endpoint",
        access_key=os.getenv("MINIO_ACCESS_KEY") or "default_access_key",
        secret_key=os.getenv("MINIO_SECRET_KEY") or "default_secret_key",
        secure=False,
    )
    data_module = LightLoader(
        batch_size,
        minio_client,
        data_name,
        seq_len,
        0,
        pred_len,
        features="M",
        target="OT",
        timeenc=0,
        freq="h",
        train_only=False,
        num_workers=4,
    )
    await ctx.report_progress(0, message="Starting training")
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, trainer.fit, model, data_module)
        await ctx.report_progress(1, message="Training completed, starting testing")
        results = await loop.run_in_executor(executor, trainer.test, model, data_module)
        await ctx.report_progress(2, message="Testing completed")
    return results


class LightNLinear(L.LightningModule):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        task_name: str,
        scheduler,
        loss: nn.Module,
        num_class: int | None = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        individual: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model = NLinear(
            task_name,
            seq_len,
            pred_len,
            channels,
            num_class=num_class,
            individual=individual,
        )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, x):
        self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = self.scheduler(optimizer, **self.kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, train_batch, batch_idx):
        batch_x, batch_y, _, _ = train_batch
        outputs = self.model(batch_x)
        loss = self.loss(outputs, batch_y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        batch_x, batch_y, _, _ = valid_batch
        outputs = self.model(batch_x)
        loss = self.loss(outputs, batch_y)

        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        batch_x, batch_y, _, _ = test_batch
        outputs = self.model(batch_x).cpu().detach().numpy()
        batch_y = batch_y.cpu().detach().numpy()
        self.log(
            "test_rmse",
            metrics.RMSE(outputs, batch_y),
        )
        self.log(
            "test_mae",
            metrics.MAE(outputs, batch_y),
        )
        self.log(
            "test_mape",
            metrics.MAPE(outputs, batch_y),
        )
        self.log(
            "test_mse",
            metrics.MSE(outputs, batch_y),
        )


class NLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        task_name: str,
        seq_len: int,
        pred_len: int,
        channels: int,
        num_class: int | None = None,
        individual=False,
    ):
        """
        individual: Bool, whether shared model among different variates.
        """
        super().__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        if (
            self.task_name == "classification"
            or self.task_name == "anomaly_detection"
            or self.task_name == "imputation"
        ):
            self.pred_len = seq_len
        else:
            self.pred_len = pred_len

        self.individual = individual
        self.channels = channels
        if self.individual:
            self.linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.linear.append(torch.nn.Linear(self.seq_len, self.pred_len))
        else:
            self.linear = torch.nn.Linear(self.seq_len, self.pred_len)

        if self.task_name == "classification":
            isinstance(num_class, int)
            self.projection = nn.Linear(self.channels * self.seq_len, num_class)  # type: ignore

    def encoder(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype, device=x.device
            )
            for i in range(self.channels):
                output[:, :, i] = self.linear[i](x[:, :, i])  # type: ignore
            x = output
        else:
            x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x += seq_last
        return x

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, **kwargs):
        if self.task_name == "forecast":
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
