import os

import hydra
import torch
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import random_split

from experts_model import OptimizerConfig, RoutingConfig, RoutingVit, create_optimizer
from process_data import test_dataset, train_dataset, val_dataset
from Trainer import Trainer, TrainerConfig


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_train_objs(
    model_cfg: RoutingConfig, opt_cfg: OptimizerConfig, device_type="cuda"
):
    model = RoutingVit(model_cfg)
    optimizer = create_optimizer(model, opt_cfg, device_type=device_type)

    return model, optimizer


@hydra.main(version_base=None, config_path=".", config_name="routing_transformer_cfg")
def main(cfg: DictConfig):
    ddp_setup()
    model_cfg = RoutingConfig(**cfg["RoutingConfig"])
    opt_cfg = OptimizerConfig(**cfg["OptimizerConfig"])
    trainer_cfg = TrainerConfig(**cfg["TrainerConfig"])
    print(model_cfg)
    model, optimizer = get_train_objs(model_cfg, opt_cfg)
    trainer = Trainer(trainer_cfg, model, optimizer, train_dataset, val_dataset)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    main()
