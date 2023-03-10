from eotorch.datasets import JSONDataset, RasterDataset
from eotorch.samplers import RandomBatchSampler, FullGridSampler
from eotorch.dataloaders import init_torch_dataloader
from eotorch.tasks import TaskTrainer, TaskMetrics, TaskLoss
from eotorch.utils import stack_samples
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, JaccardIndex
from pytorch_lightning import Trainer
from pathlib import Path

import segmentation_models_pytorch as smp

bands = [1, 2, 3, 7]
label_column = "label_id"
label_ids = [0, 1]


class Loss(TaskLoss):
    @staticmethod
    def preprocessing(y, y_hat):
        return y.unsqueeze(1).float(), y_hat


class Metrics(TaskMetrics):
    @staticmethod
    def preprocessing(y: torch.Tensor, y_hat: torch.Tensor):
        return y.unsqueeze(1).float(), y_hat.sigmoid().float()


print("train dataloader")

image_dataset = RasterDataset(
    root="~/Data/intelligon/train",
    bands=bands,
    dtype=torch.float,
)
label_dataset = JSONDataset(
    root="~/Data/intelligon/train",
    label_column=label_column,
    indata=label_ids,
    dtype=torch.float,
)
dataset = image_dataset & label_dataset
sampler = RandomBatchSampler(
    size=128,
    batch_size=10,
    length=100,
)
train_dataloader = init_torch_dataloader(
    dataset=dataset,
    sampler=sampler,
    collate_fn=stack_samples,
)

print("val dataloader")

image_dataset = RasterDataset(
    root="~/Data/intelligon/val",
    bands=bands,
    dtype=torch.float,
)
label_dataset = JSONDataset(
    root="~/Data/intelligon/val",
    label_column=label_column,
    indata=label_ids,
    dtype=torch.float,
)
dataset = image_dataset & label_dataset
sampler = FullGridSampler(
    size=128,
    stride=128,
)
val_dataloader = init_torch_dataloader(
    dataset=dataset,
    sampler=sampler,
    batch_size=10,
    collate_fn=stack_samples,
)

print("model")

model = smp.UnetPlusPlus(
    encoder_name="resnet18",
    decoder_use_batchnorm=False,
    in_channels=len(bands),
    classes=1,
)

print("loss")

loss = Loss(BCEWithLogitsLoss())

print("metrics")

metrics = Metrics(
    [
        Accuracy(task="binary"),
        JaccardIndex(task="binary"),
    ],
)

print("optimizer")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_config = {
    "optimizer": optimizer,
    "lr_scheduler": {
        "scheduler": ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=10,
        ),
        "monitor": "val_BinaryAccuracy",
    },
}

print("task")

task = TaskTrainer(
    model=model,
    loss=loss,
    metrics=metrics,
    optimizer_config=optimizer_config,
)

print("trainer")

trainer = Trainer(
    max_epochs=1,
    default_root_dir=str(Path.home().resolve()),
)

print("fit")

trainer.fit(
    model=task,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
