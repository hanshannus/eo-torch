from typing import Any, Dict, Callable, Union, Sequence

import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.metric import Metric


class _TaskStatus:
    TRAINING = "train"
    VALIDATING = "val"
    TESTING = "test"


class TaskLoss(nn.Module):
    def __init__(self, loss=None):
        super().__init__()
        self.loss = loss

    def forward(self, y, y_hat) -> Tensor:
        y, y_hat = self.preprocessing(y, y_hat)
        return self.loss(y_hat, y)

    @staticmethod
    def preprocessing(y, y_hat):
        return y, y_hat


class TaskMetrics:
    def __init__(
            self,
            metrics: Union[MetricCollection, Metric, Sequence[Metric], Dict[str, Metric]],
            val_metrics: nn.Module = None,
            test_metrics: nn.Module = None,
            train_metrics: nn.Module = None,
    ):
        super().__init__()
        if metrics is None and train_metrics is None:
            raise
        if metrics is not None and train_metrics is not None:
            raise
        self.train_metrics = metrics if metrics is not None else train_metrics
        if not isinstance(self.train_metrics, MetricCollection):
            self.train_metrics = MetricCollection(self.train_metrics)
        # infer validation and test metric from training metric if not provided
        self.train_metrics = self.train_metrics.clone(prefix="train_")
        if val_metrics is None:
            val_metrics = self.train_metrics.clone(prefix="val_")
        self.val_metrics = val_metrics
        if test_metrics is None:
            test_metrics = self.train_metrics.clone(prefix="test_")
        self.test_metrics = test_metrics

    @staticmethod
    def preprocessing(y, y_hat):
        return y, y_hat

    def update(self, y, y_hat, status: str):
        y, y_hat = self.preprocessing(y, y_hat)
        if status == _TaskStatus.TRAINING:
            self.train_metrics(y_hat, y)
        elif status == _TaskStatus.VALIDATING:
            self.val_metrics(y_hat, y)
        elif status == _TaskStatus.TESTING:
            self.test_metrics(y_hat, y)
        else:
            raise ValueError(f"Status '{status}' is not defined.")

    def compute(self, status: str):
        if status == _TaskStatus.TRAINING:
            return self.train_metrics.compute()
        if status == _TaskStatus.VALIDATING:
            return self.val_metrics.compute()
        if status == _TaskStatus.TESTING:
            return self.test_metrics.compute()
        raise ValueError(f"Status '{status}' is not defined.")

    def reset(self, status: str):
        if status == _TaskStatus.TRAINING:
            self.train_metrics.reset()
        elif status == _TaskStatus.VALIDATING:
            self.val_metrics.reset()
        elif status == _TaskStatus.TESTING:
            self.test_metrics.reset()
        else:
            raise ValueError(f"Status '{status}' is not defined.")


class TaskPlotter:
    def __init__(
            self,
            train_plotter: nn.Module = None,
            val_plotter: nn.Module = None,
            test_plotter: nn.Module = None,
    ):
        super().__init__()
        self.train_plotter = train_plotter
        self.val_plotter = val_plotter
        self.test_plotter = test_plotter

    @staticmethod
    def preprocessing(self, x, y, y_hat):
        return x, y, y_hat

    def compute(self, x, y, y_hat, status: str):
        x, y, y_hat = self.preprocessing(x, y, y_hat)
        if status == _TaskStatus.TRAINING and self.train_plotter is not None:
            self.train_plotter(x, y, y_hat)
        if status == _TaskStatus.VALIDATING and self.val_plotter is not None:
            self.val_plotter(x, y, y_hat)
        if status == _TaskStatus.TESTING and self.test_plotter is not None:
            self.test_plotter(x, y, y_hat)


class TaskTrainer(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            loss: TaskLoss,
            metrics: TaskMetrics,
            optimizer_config,
            plotter: TaskPlotter = None,
            preprocessing: Callable[[Tensor], Any] = None,
            postprocessing: Callable[[Any], Tensor] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.loss = loss
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.metrics = metrics
        self.plotter = plotter
        self._optimizer_config = optimizer_config

    def forward(
            self,
            x: Tensor,
            *args: Any,
            **kwargs: Any,
    ) -> Tensor:
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        y_hat = self.model(x, *args, **kwargs)
        if self.postprocessing is not None:
            y_hat = self.postprocessing(y_hat)
        return y_hat

    def step(
            self,
            batch: Dict[str, Tensor],
            status: str,
            *args: Any,
            **kwargs: Any,
    ) -> Tensor:
        x, y = batch["image"], batch["mask"]
        y_hat = self.forward(x, *args, **kwargs)
        if self.plotter is not None:
            self.plotter.compute(x, y, y_hat, status)
        self.metrics.update(y, y_hat, status)
        return self.loss(y, y_hat)

    def epoch_end(self, status: str):
        self.log_dict(
            self.metrics.compute(status),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.metrics.reset(status)

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.step(batch, _TaskStatus.TRAINING)

    def training_epoch_end(self, outputs: Any) -> None:
        self.epoch_end(_TaskStatus.TRAINING)

    def validation_step(self, batch: Dict[str, Tensor], batch_idx) -> Tensor:
        return self.step(batch, _TaskStatus.VALIDATING)

    def validation_epoch_end(self, outputs: Any) -> None:
        self.epoch_end(_TaskStatus.VALIDATING)

    def test_step(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.step(batch, _TaskStatus.TESTING)

    def test_epoch_end(self, outputs: Any) -> None:
        self.epoch_end(_TaskStatus.TESTING)

    def configure_optimizers(self) -> Dict[str, Any]:
        return self._optimizer_config
