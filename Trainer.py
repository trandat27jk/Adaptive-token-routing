import math
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import fsspec
import torch
import torch.amp
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class Snapshot:
    model_state: "OrderedDict[str, torch.Tensor]"
    optimizer_state: Dict[str, Any]
    finished_epoch: int


@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    save_every: int = None
    use_amp: bool = True
    snapshot_path: Optional[str] = None
    warmup: int = 20
    max_lr: float = 2e-2
    min_lr: float = 4e-3


class Trainer(nn.Module):
    def __init__(
        self,
        trainer_config: TrainerConfig,
        model,
        optimizer,
        train_dataset,
        test_dataset=None,
    ):
        super().__init__()
        self.trainer_config = trainer_config
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = (
            self._prepare_dataloader(test_dataset) if test_dataset else None
        )
        self.optimizer = optimizer
        self.model = model.to(self.local_rank)
        if self.trainer_config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.trainer_config.snapshot_path is None:
            self.trainer_config.snapshot_path = "snapshot.pt"
        self.epochs_run = 0

        self._load_snapshot()
        self.train_loss = {}
        self.val_loss = {}
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.total_correct = 0
        self.total_samples = 0

    def _run_batch(self, source, target, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.trainer_config.use_amp),
        ):
            output = self.model(source)
            loss = F.cross_entropy(output, target)

        if train:
            self.optimizer.zero_grad()
            if self.trainer_config.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.trainer_config.grad_norm_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.trainer_config.grad_norm_clip
                )
                self.optimizer.step()
        else:
            prediction = F.softmax(output, dim=-1)
            prediction = torch.argmax(prediction, dim=-1)
            self.total_correct += (prediction == target).sum().item()
            self.total_samples += target.size(0)

        return loss.item()

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.trainer_config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")

        except FileNotFoundError:
            print("not found. Train from scratch")
            return

        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot_data.model_state)
        self.optimizer.load_state_dict(snapshot_data.optimizer_state)
        self.epochs_run = snapshot_data.finished_epoch
        print(f"Resuming at epochs{self.epochs_run}")

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=False,
            num_workers=self.trainer_config.data_loader_workers,
            sampler=DistributedSampler(dataset),
        )

    # cosineAnnelingLr
    def _get_lr(self, epoch):
        if epoch < self.trainer_config.warmup:
            lr = self.trainer_config.max_lr * (
                (epoch + 1) / (self.trainer_config.warmup + 1)
            )
        elif epoch >= self.trainer_config.max_epochs:
            lr = self.trainer_config.min_lr
        else:
            cur_gap = epoch - self.trainer_config.warmup
            max_gap = self.trainer_config.max_epochs - self.trainer_config.warmup

            lr = self.trainer_config.min_lr + (1 / 2) * (
                self.trainer_config.max_lr - self.trainer_config.min_lr
            ) * (1 + math.cos((cur_gap / max_gap) * math.pi))
        return lr

    def _run_epoch(self, epoch, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        for batch_idx, (source, target) in enumerate(dataloader):
            source = source.to(self.local_rank)
            target = target.to(self.local_rank)
            loss = self._run_batch(source, target, train)
            if train:
                self.train_loss[f"{epoch}"] = loss
                print(
                    f"[GPU{self.global_rank}] Epoch {epoch} |Iter {batch_idx} | loss{loss}"
                )
            else:
                self.val_loss[f"{epoch}"] = loss
                print("_____evaluation_______")
                print(
                    f"[GPU{self.global_rank}] Epoch {epoch} |Iter {batch_idx} | val loss{loss}"
                )

    def _save_snapshot(self, epoch):
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer,
            finished_epoch=epoch,
        )

        torch.save(snapshot, self.trainer_config.snapshot_path)

        print(f"snapshot is saved at {epoch}")

    def train(self):
        for epoch in range(self.trainer_config.max_epochs):
            lr = self._get_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            epoch += 1
            self._run_epoch(epoch, self.train_loader, train=True)
            if epoch % self.trainer_config.save_every == 0:
                self._save_snapshot(epoch)
                self.model.eval()
                self.total_correct = 0
                self.total_samples = 0
                with torch.no_grad():
                    self._run_epoch(epoch, self.test_loader, train=False)
                    accuracy = self.total_correct / self.total_samples
                    print(f"val accuracy :{accuracy}")

        with open("training_loss.txt", "w") as f:
            for key, value in self.train_loss.items():
                f.write(f"{key}:{value}\n")
        with open("vaidation_loss.txt", "w") as f:
            for key, value in self.train_loss.items():
                f.write(f"{key}:{value}\n")
