import math
import os
import time
from contextlib import nullcontext

import fsspec
import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from experts_model import OptimizerConfig, RoutingConfig, RoutingVit
from process_data import test_dataset, train_dataset

torch._dynamo.config.suppress_errors = True

# wandb
wandb_log = False
wandb_project = "Routing Transformer"
wandb_run_name = "Routing Patch 7"
# RoutingConfig:
image_size = 224
patch_size = 14
n_layers = 12
embedd_dim = 512
n_heads = 8
factor = 4
channels = 3
top_k = 128
bias = False
classes = 2

# OptimizerConfig:
decay_lr = True
learning_rate = 3e-4
weight_decay = 0.1
betas = [0.9, 0.999]

# TrainerConfig:
max_epochs = 224
batch_size = 32
data_loader_workers = 2
grad_norm_clip = 1.0
save_every = 5
use_amp = True
snapshot_path = "routing_snapshot.pt"
init_from = "scratch"
out_dir = "out"
warmup = 20
max_lr = 2e-2
min_lr = 4e-3


def compute_iou(y_true, y_pred, num_classes):
    iou_per_class = []

    for c in range(num_classes):
        intersection = torch.logical_and(y_true == c, y_pred == c).sum().item()
        union = torch.logical_or(y_true == c, y_pred == c).sum().item()

        if union == 0:
            iou_per_class.append(float("nan"))
        else:
            iou_per_class.append(intersection / union)

    return np.nanmean(iou_per_class)


# ddp settings
backend = "nccl"

device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True
# ------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}


ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)

model_args = dict(
    image_size=image_size,
    patch_size=patch_size,
    n_layers=n_layers,
    embedd_dim=embedd_dim,
    n_heads=n_heads,
    factor=factor,
    channels=channels,
    top_k=top_k,
    bias=bias,
    classes=classes,
)

if init_from == "scratch":
    print("Initializing a new model from scratch")

    RoutingConfig = RoutingConfig(**model_args)
    model = RoutingVit(RoutingConfig)

elif init_from == "resume":
    try:
        snapshot = fsspec.open(os.path.join(out_dir, snapshot_path))
        with snapshot as f:
            snapshot_data = torch.load(f, map_location=device)
    except FileNotFoundError:
        print("training from scratch")

    snapshot_args = snapshot_data["model_args"]
    for k in [
        "image_size",
        "patch_size",
        "n_layers",
        "embedd_dim",
        "n_heads",
        "factor",
        "channels",
        "top_k",
        "bias",
        "classes",
    ]:
        model_args[k] = snapshot_args[k]
    RoutingConfig = RoutingConfig(**model_args)
    model = RoutingVit(RoutingConfig)
    state_dict = snapshot_data["model"]
    model.load_state_dict(state_dict)
    iter_num = snapshot_data["iter_num"]
    best_val_loss = snapshot_data["best_val_loss"]


scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
model.to(device)
optimizer = model.create_optimizer(
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    betas=betas,
    device_type=device,
)
if init_from == "resume":
    optimizer.load_state_dict(snapshot_data["optimizer"])
snapshot_data = None

if compile:
    print("using compile")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


def get_lr(it):
    if it < warmup:
        return learning_rate * (it + 1) / (warmup + 1)
    if it > max_epochs:
        return min_lr

    decay_ratio = (it - warmup) / (max_epochs - warmup)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# process data
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
)
best_accuray = 0
raw_model = model.module if ddp else model


for i in range(max_epochs):
    loss_total_train = 0
    train_total_samples = 0
    lr = get_lr(i) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    for batch_idx, (source, target) in enumerate(train_loader):
        source, target = source.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            output = model(source)
            loss = F.cross_entropy(output, target)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_total_train += loss.detach() * source.size(0)
        train_total_samples += source.size(0)
    loss_train = loss_total_train / train_total_samples
    if master_process:
        print(f"loss_train {loss_train} at epoch {i}")

    if i % save_every == 0 and master_process:
        with torch.no_grad():
            loss_total = 0
            loss_val = 0
            accuracy = 0
            sample = 0
            corrects = 0
            model.eval()
            for val_idx, (source, target) in enumerate(test_loader):
                source, target = source.to(device, non_blocking=True), target.to(
                    device, non_blocking=True
                )
                output = model(source)
                loss = F.cross_entropy(output, target)
                predictions = torch.argmax(output, dim=1)
                corrects += (predictions == target).sum().item()
                loss_total += loss.item() * source.size(0)
                sample += source.size(0)

            loss_val = loss_total / sample
            accuracy = corrects / sample
        if wandb_log:
            wandb_log(
                {
                    "epoch": i,
                    "train_loss": loss_train,
                    "val_loss": loss_val,
                    "lr": lr,
                }
            )
        if accuracy > best_accuray:
            best_accuray = accuracy
            snapshot_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "epoch": i,
                "best_accuracy": best_accuray,
                "config": config,
            }
            if master_process:
                print(f"save model at epoch {i}")
                print(f"accuracy {accuracy} at epoch {i}")
            torch.save(snapshot_data, os.path.join(out_dir, "snapshot.pt"))


if ddp:
    destroy_process_group()
