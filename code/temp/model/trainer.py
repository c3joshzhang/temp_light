import os
import time
from pathlib import Path
from pprint import pprint

import wandb
from sklearn.model_selection import ParameterGrid
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from temp.deprecate.global_vars import *
from temp.model.loss import scoring
from temp.model.utils import *

EPS = torch.tensor(1e-8).to(DEVICE)
zero = torch.tensor(0).to(DEVICE)
one = torch.tensor(1).to(DEVICE)
GLOBAL_STEP = None


def pretrain(model, pretrain_loader):
    model.pre_train_init()
    i = 0
    while True:
        for batch_idx, (graph_idx, batch) in enumerate(tqdm(pretrain_loader)):
            batch.to(DEVICE)
            if not model.pre_train(batch):
                break

        if model.pre_train_next() is None:
            break
        i += 1
    return i


def step(
    epoch,
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    step_type,
    bias_threshold=0.5,
    binary_pred=False,
    eval=True,
    print_log=True,
    **kwargs,
):

    global GLOBAL_STEP
    model.eval() if eval else model.train()

    if step_type == "train" and not eval:
        print("Training...", GLOBAL_STEP)
    elif step_type == "train" and eval:
        print("Evaluating training set...")
    elif step_type == "val":
        print("Validating...")

    loss_all = torch.tensor(0.0).to(DEVICE)
    bias_tuples = []
    evidence_tuples = []

    for batch_idx, (graph_idx, batch) in enumerate(tqdm(loader)):
        free_gpu_memory()

        batch = batch.to(DEVICE)
        y = torch.where(batch.y_incumbent <= bias_threshold, zero, one).to(DEVICE)

        output = model(batch)

        loss, evidence_tuple, uncertainty = criterion(
            GLOBAL_STEP, graph_idx, batch, output, y, binary_pred, step_type
        )

        if not eval:
            loss.backward()
            optimizer.step()

            if type(scheduler) in [NoamLR, lr_scheduler.OneCycleLR]:
                scheduler.step()

            optimizer.zero_grad()

            GLOBAL_STEP += 1

        loss_all += loss.item()

        evidence_tuples.append(evidence_tuple)

        pred = torch.softmax(output, dim=-1)[:, 1].view(-1)

        if binary_pred:
            pred = pred[batch.is_binary]
            y = y[batch.is_binary]
            # uncertainty = uncertainty[batch.is_binary]

        pred_all = torch.cat([pred_all, pred]) if batch_idx > 0 else pred
        y_all = torch.cat([y_all, y]) if batch_idx > 0 else y
        uncertainty_all = (
            torch.cat([uncertainty_all, uncertainty]) if batch_idx > 0 else uncertainty
        )

        true_bias = torch.mean(y.to(torch.float))
        pred_bias = torch.mean(pred.round())
        soft_pred_bias = torch.mean(pred)
        bias_tuples.append(
            [true_bias, pred_bias, soft_pred_bias, torch.abs(true_bias - pred_bias)]
        )

    if not eval and type(scheduler) is lr_scheduler.StepLR:
        scheduler.step()

    lr = scheduler.optimizer.param_groups[0]["lr"]
    log = scoring(
        batch_idx,
        loss_all,
        bias_tuples,
        evidence_tuples,
        uncertainty_all,
        pred_all,
        y_all,
        lr,
        bias_threshold,
        step_type,
        print_log,
    )

    return log


def train(
    model_name,
    model,
    criterion,
    optimizer,
    scheduler,
    pretrain_loader,
    train_loader,
    val_loader,
    config,
    WANDB_LOG=False,
    model_dir="",
):

    global GLOBAL_STEP
    GLOBAL_STEP = 0

    print(">> Training starts on the current device", DEVICE)

    if WANDB_LOG:
        run = wandb.init(project="gnn4co", config=config, force=True, name=model_name)

    if config["prenorm"]:
        print(">> Pretraining for prenorm...")
        pretrain(model, pretrain_loader)

    for epoch in range(1, config["num_epochs"] + 1):
        print(">> Epoch", epoch, "".join(["-"] * 100))
        epoch_time = time.time()

        train_log = step(
            epoch,
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            "train",
            eval=False,
            **config,
        )

        with torch.no_grad():
            free_gpu_memory()
            val_log = step(
                epoch,
                model,
                val_loader,
                optimizer,
                scheduler,
                criterion,
                "val",
                eval=True,
                **config,
            )

            if epoch == 1 or round(val_log["val_acc"], 3) >= best_val_score:

                best_val_score = round(val_log["val_acc"], 3)

                # if model_dir:
                #     torch.save(model.state_dict(), str(model_dir.joinpath(model_name + ".pt")))

        epoch_time = time.time() - epoch_time
        log = {"epoch": epoch, "epoch_time": epoch_time, **train_log, **val_log}

        if type(scheduler) is lr_scheduler.ReduceLROnPlateau:
            scheduler.step(val_log["val_acc"])

        if WANDB_LOG:
            wandb.log(log)

    if WANDB_LOG:
        run.finish()

    return model
