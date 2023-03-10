from eotorch.datasets import JSONDataset, RasterDataset
from eotorch.samplers import RandomBatchSampler, FullGridSampler
from eotorch.dataloaders import init_torch_dataloader
from eotorch.tasks import TaskTrainer, TaskMetrics, TaskLoss
from eotorch.utils import stack_samples
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, JaccardIndex
from pytorch_lightning import Trainer
from pathlib import Path

from transformers.models.maskformer import (
    MaskFormerForInstanceSegmentation,
    MaskFormerConfig,
)
from transformers.models.maskformer.modeling_maskformer import (
    MaskFormerForInstanceSegmentationOutput,
)
from transformers.models.swin import SwinConfig
from transformers.models.detr import DetrConfig


bands = [1, 2, 3, 7]
label_column = "label_id"
label_ids = [0, 1, 2]
id2label = {0: "bg", 1: "pond", 2: "channel"}
label2id = {"bg": 0, "pond": 1, "channel": 2}


class Loss(TaskLoss):
    @staticmethod
    def preprocessing(y, y_hat):
        return y.long(), y_hat.float()


class Metrics(TaskMetrics):
    @staticmethod
    def preprocessing(y: torch.Tensor, y_hat: torch.Tensor):
        return y.float(), y_hat.argmax(dim=1).float()


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

model = MaskFormerForInstanceSegmentation(
    MaskFormerConfig(
        backbone_config=SwinConfig(
            num_channels=4, id2label=id2label, label2id=label2id
        ).to_dict(),
        decoder_config=DetrConfig(
            num_channels=4, id2label=id2label, label2id=label2id
        ).to_dict(),
    )
)
model.training = True


def postprocessing(outputs: MaskFormerForInstanceSegmentationOutput):
    # global image_processor
    target_sizes = [128, 128]
    # Remove the null class `[..., :-1]`
    masks_classes = outputs.class_queries_logits.softmax(dim=-1)
    masks_probs = outputs.masks_queries_logits.sigmoid()
    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    # rescale
    return torch.nn.functional.interpolate(
        segmentation, size=target_sizes, mode="bilinear", align_corners=False
    )


print("loss")

loss = Loss(CrossEntropyLoss())

print("metrics")

metrics = Metrics(
    [
        Accuracy(task="multiclass", num_classes=len(label_ids)),
        JaccardIndex(task="multiclass", num_classes=len(label_ids)),
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
        "monitor": "val_MulticlassAccuracy",
    },
}

print("task")

task = TaskTrainer(
    model=model,
    loss=loss,
    metrics=metrics,
    optimizer_config=optimizer_config,
    postprocessing=postprocessing,
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
