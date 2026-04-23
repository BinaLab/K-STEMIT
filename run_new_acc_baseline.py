"""Training entry point for standalone GAT and GIN baseline comparison experiments."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import StepLR
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model_baseline_0419 import IceSheetModel_Multibranch_ablation4_GAT
from model_baseline_0419 import IceSheetModel_Multibranch_ablation4_GIN
from model_baseline_0419 import IceSheetModel_Multibranch_ablation4_nonphy_GAT
from model_baseline_0419 import IceSheetModel_Multibranch_ablation4_nonphy_GIN
from utils import PolyLR
from utils import load_dill
from utils import rmse
from utils import rmse_dim
from utils import save_model
from utils import split_dataset


def _default_dataset_path():
    home_dataset = Path.home() / "Internal_Ice_Layer_GNN_Dataset/Shallow-To-Deep/mixture/Shallow-To-Deep_Mix_WithPhysics/dataset"
    legacy_dataset = Path("/home/zel220/Internal_Ice_Layer_GNN_Dataset/Shallow-To-Deep/mixture/Shallow-To-Deep_Mix_WithPhysics/dataset")
    for candidate in (home_dataset, legacy_dataset):
        if candidate.is_file():
            return str(candidate)
    return str(home_dataset)


def _default_split_file():
    home_split = Path.home() / "Internal_Ice_Layer_GNN_Dataset/Shallow-To-Deep/mixture/Shallow-To-Deep_Mix_WithPhysics/splits.npy"
    legacy_split = Path("/home/zel220/Internal_Ice_Layer_GNN_Dataset/Shallow-To-Deep/mixture/Shallow-To-Deep_Mix_WithPhysics/splits.npy")
    for candidate in (home_split, legacy_split):
        if candidate.is_file():
            return str(candidate)
    return str(home_split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone baseline training for the 0419 GATConv/GINConv multi-branch models.")
    parser.add_argument("--model", type=str, default="GAT", help="Baseline model name: GAT or GIN")
    parser.add_argument("--batch", type=int, default=1, help="Batch Size")
    parser.add_argument("--epoch", type=int, default=300, help="Epoch")
    parser.add_argument("--adaptive", default="False", help="Use Adaptive")
    parser.add_argument("--ablation", default="False", help="Run physical feature ablation")
    parser.add_argument("--nonphysical", "--non-physical", dest="nonphysical", default="False", help="Strip inputs down to the first 3 non-physical features")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    parser.add_argument("--featureablation", default="0000000", help="7-bit physical feature ablation mask")
    parser.add_argument("--folder", default="Experiment", help="Folder Name For Saving")
    parser.add_argument("--scheduler", default="poly", help="Learning Rate Scheduler")
    parser.add_argument("--schedulerargs", type=float, help="Learning Rate Scheduler Argument")
    parser.add_argument("--eta_min", type=float, default=1e-7, help="Minimum Learning Rate for Cosine Annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight Decay for Optimizer")
    parser.add_argument("--dataset-path", default=_default_dataset_path(), help="Dataset dill path")
    parser.add_argument("--split-file", default=_default_split_file(), help="Dataset split file path")
    return parser.parse_args()


def build_scheduler(args_cml, optimizer, arg):
    if args_cml.scheduler == "poly":
        return PolyLR(optimizer, max_iters=arg.epoch, power=args_cml.schedulerargs)
    if args_cml.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=int(args_cml.schedulerargs), eta_min=arg.eta_min)
    if args_cml.scheduler == "warmup":
        return CosineAnnealingWarmRestarts(optimizer, T_0=int(args_cml.schedulerargs), eta_min=1e-6)
    if args_cml.scheduler == "exp":
        return ExponentialLR(optimizer, gamma=0.99)
    if args_cml.scheduler == "step":
        return StepLR(optimizer, step_size=int(args_cml.schedulerargs), gamma=0.5)
    if args_cml.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.33,
            patience=int(args_cml.schedulerargs),
            threshold=1e-6,
            cooldown=5,
            verbose=True,
            min_lr=1e-6,
        )
    raise ValueError(f"Unsupported scheduler: {args_cml.scheduler}")


def validate_feature_ablation(feature_ablation):
    if len(feature_ablation) != 7 or any(bit not in {"0", "1"} for bit in feature_ablation):
        raise ValueError("featureablation must be a 7-character bitmask made of 0 and 1.")


def validate_dataset(dataset_path, split_file, remove_all_physical_params):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    dataset = load_dill(dataset_path)
    if len(dataset) == 0 or len(dataset[0]) == 0:
        raise ValueError(f"Dataset is empty or malformed: {dataset_path}")

    feature_count = dataset[0][0].x.shape[1]
    if feature_count not in (3, 10):
        raise ValueError(f"Unexpected dataset feature count {feature_count}; expected 3 or 10.")
    if not remove_all_physical_params and feature_count != 10:
        raise ValueError(
            f"Physical baseline training expects 10 input features, but dataset provides {feature_count}."
        )
    if remove_all_physical_params and feature_count < 3:
        raise ValueError(f"Non-physical training expects at least 3 input features; found {feature_count}.")

    return dataset, feature_count


def strip_to_nonphysical_features(dataset_split):
    for graph_collection in dataset_split:
        for graph in graph_collection:
            if graph.x.shape[1] > 3:
                graph.x = graph.x[:, :3].clone()


def apply_feature_ablation(dataset_split, feature_ablation):
    for graph_collection in dataset_split:
        for graph in graph_collection:
            if graph.x.shape[1] <= 3:
                continue
            for feature_idx, bit in enumerate(feature_ablation):
                if bit == "0":
                    graph.x[:, 3 + feature_idx] = 0


def normalize_dataset(dataset_split, train_features_mean, train_features_std):
    for graph_collection in dataset_split:
        for graph in graph_collection:
            graph.x -= train_features_mean
            graph.x /= train_features_std


def build_model(args_cml, arg, accelerator):
    model_name = args_cml.model.upper()
    if model_name == "GAT" and arg.NON_PHYSICAL:
        return IceSheetModel_Multibranch_ablation4_nonphy_GAT(arg).to(accelerator.device)
    if model_name == "GIN" and arg.NON_PHYSICAL:
        return IceSheetModel_Multibranch_ablation4_nonphy_GIN(arg).to(accelerator.device)
    if model_name == "GAT":
        return IceSheetModel_Multibranch_ablation4_GAT(arg).to(accelerator.device)
    if model_name == "GIN":
        return IceSheetModel_Multibranch_ablation4_GIN(arg).to(accelerator.device)
    raise ValueError(f"Unsupported baseline model: {args_cml.model}")


def main():
    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)
    set_seed(seed)

    args_cml = parse_args()
    validate_feature_ablation(args_cml.featureablation)

    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(device_placement=True, step_scheduler_with_optimizer=False, dataloader_config=dataloader_config)
    accelerator.print("Args:", args_cml)
    device = accelerator.device

    nonphysical = args_cml.nonphysical.lower() == "true"
    run_ablation_study = args_cml.ablation.lower() == "true" and not nonphysical

    dataset, source_feature_count = validate_dataset(
        os.path.abspath(os.path.expanduser(args_cml.dataset_path)),
        os.path.abspath(os.path.expanduser(args_cml.split_file)),
        nonphysical,
    )

    arg = SimpleNamespace()
    arg.FEATURE_ABLATION = args_cml.featureablation
    arg.NON_PHYSICAL = nonphysical
    arg.REMOVE_ALL_PHYSICAL_PARAMS = nonphysical
    arg.NODE_COUNT = 256
    arg.FEATURE_COUNT = 3 if nonphysical else 10
    arg.EXPERIMENT_NAME = args_cml.folder
    arg.LAYER_PREDICT_COUNT = 15
    arg.LAYER_FEATURE_COUNT = 5
    arg.PREDICT_HISTORIC = False
    arg.ADAPTIVE = args_cml.adaptive.lower() == "true"
    arg.DIMENSIONALITIES = [256, 128, 64]
    arg.INITIAL_LEARNING_RATE = args_cml.lr
    arg.eta_min = args_cml.eta_min
    arg.weight_decay = args_cml.weight_decay
    arg.RUN_ABLATION_STUDY = run_ablation_study
    arg.BATCH_SIZE = args_cml.batch
    arg.epoch = args_cml.epoch
    arg.SPLIT_FILE = os.path.abspath(os.path.expanduser(args_cml.split_file))

    os.makedirs(arg.EXPERIMENT_NAME, exist_ok=True)

    accelerator.print("Resolved Dataset Path:", os.path.abspath(os.path.expanduser(args_cml.dataset_path)))
    accelerator.print("Resolved Split File:", arg.SPLIT_FILE)
    accelerator.print("Source Feature Count:", source_feature_count)
    accelerator.print("Training Feature Count:", arg.FEATURE_COUNT)
    accelerator.print("Non-Physical Mode:", nonphysical)
    accelerator.print("Run Ablation Study:", arg.RUN_ABLATION_STUDY)

    for split_idx in range(5):
        SPLIT = split_idx

        accelerator.print("Splitting Dataset")
        accelerator.print("Split Number: ", SPLIT)
        training_dataset, val_dataset, testing_dataset = split_dataset(dataset, SPLIT, arg.SPLIT_FILE)
        accelerator.print("Training: ", len(training_dataset))
        accelerator.print("Validation: ", len(val_dataset))
        accelerator.print("Testing: ", len(testing_dataset))

        if arg.REMOVE_ALL_PHYSICAL_PARAMS:
            accelerator.print("Removing physical features before normalization")
            strip_to_nonphysical_features(training_dataset)
            strip_to_nonphysical_features(val_dataset)
            strip_to_nonphysical_features(testing_dataset)

        accelerator.print("Normalizing Datasets")
        all_features = None
        for graph_collection in tqdm(training_dataset, disable=not accelerator.is_local_main_process):
            for graph in graph_collection:
                if all_features is None:
                    all_features = graph.x
                else:
                    all_features = torch.cat((all_features, graph.x), 0)

        train_features_mean = torch.mean(all_features, dim=0)
        train_features_std = torch.std(all_features, dim=0)
        train_features_std[train_features_std == 0] = 1
        accelerator.print("Normalized means and stds:")
        accelerator.print(train_features_mean)
        accelerator.print(train_features_std)

        normalize_dataset(training_dataset, train_features_mean, train_features_std)
        normalize_dataset(val_dataset, train_features_mean, train_features_std)
        normalize_dataset(testing_dataset, train_features_mean, train_features_std)

        if arg.RUN_ABLATION_STUDY:
            accelerator.print("Applying physical feature ablation mask:", arg.FEATURE_ABLATION)
            apply_feature_ablation(training_dataset, arg.FEATURE_ABLATION)
            apply_feature_ablation(val_dataset, arg.FEATURE_ABLATION)
            apply_feature_ablation(testing_dataset, arg.FEATURE_ABLATION)

        train_dataloader = DataLoader(training_dataset, shuffle=False, batch_size=arg.BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=arg.BATCH_SIZE)
        test_dataloader = DataLoader(testing_dataset, shuffle=False, batch_size=arg.BATCH_SIZE)
        accelerator.print("Train Loader", len(train_dataloader))
        accelerator.print("Val Loader", len(val_dataloader))
        accelerator.print("Test Loader", len(test_dataloader))

        model = build_model(args_cml, arg, accelerator)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.INITIAL_LEARNING_RATE, weight_decay=arg.weight_decay)
        lr_scheduler = build_scheduler(args_cml, optimizer, arg)

        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader)

        accelerator.print("Everything is Ready! Let's train our model!")

        TRAINING_LOSSES = []
        VALIDATION_LOSSES = []
        TRAINING_RMSE = []
        VALIDATION_RMSE = []
        TEST_RMSE = []

        best_val_loss = 999999
        best_model_epoch = 0
        seed_everything(1337)
        set_seed(1337)

        for epoch in tqdm(range(arg.epoch), disable=not accelerator.is_local_main_process):
            tr_all_preds = np.zeros((len(train_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT))
            tr_all_reals = np.zeros((len(train_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT))

            tr_index_offset = 0
            total_loss = 0

            model.train()
            for batch in train_dataloader:
                batch = [comp.to(accelerator.device) for comp in batch]
                optimizer.zero_grad()

                pred = model(batch)
                real = batch[0].y

                loss = F.mse_loss(pred.float(), real.float())
                total_loss += loss.item()

                tr_all_preds[tr_index_offset:tr_index_offset + len(pred)] = pred.cpu().detach().numpy()
                tr_all_reals[tr_index_offset:tr_index_offset + len(real)] = real.cpu().detach().numpy()

                accelerator.backward(loss)
                optimizer.step()

                tr_index_offset += arg.NODE_COUNT * arg.BATCH_SIZE

            accelerator.wait_for_everyone()

            train_rmse = rmse(np.array(tr_all_preds), np.array(tr_all_reals))
            total_loss = total_loss / len(train_dataloader)
            accelerator.print(f"Epoch #{epoch} Device #{device} Training RMSE: ", train_rmse, "; Loss: ", total_loss)
            TRAINING_LOSSES.append(total_loss)
            TRAINING_RMSE.append(train_rmse)

            val_all_preds = np.zeros((len(val_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT))
            val_all_reals = np.zeros((len(val_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT))
            model.eval()

            with torch.no_grad():
                total_val_loss = 0
                val_index_offset = 0

                for batch in val_dataloader:
                    batch = [comp.to(accelerator.device) for comp in batch]
                    pred = model(batch)
                    real = batch[0].y

                    val_loss = F.mse_loss(pred.float(), real.float())
                    total_val_loss += val_loss

                    val_all_preds[val_index_offset:val_index_offset + len(pred)] = pred.cpu().detach().numpy()
                    val_all_reals[val_index_offset:val_index_offset + len(real)] = real.cpu().detach().numpy()

                    val_index_offset += arg.NODE_COUNT * arg.BATCH_SIZE

            accelerator.wait_for_everyone()

            model.train()
            val_rmse = rmse(np.array(val_all_preds), np.array(val_all_reals))
            total_val_loss = total_val_loss / len(val_dataloader)
            accelerator.print(f"Epoch #{epoch} Validation RMSE: ", val_rmse, "; Loss: ", total_val_loss.item())
            VALIDATION_RMSE.append(val_rmse)
            VALIDATION_LOSSES.append(total_val_loss.item())

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_model_epoch = epoch
                save_model(model, arg, SPLIT, accelerator)
                accelerator.print("Best model! Val loss = ", best_val_loss)
            accelerator.print("Best epoch so far: ", best_model_epoch)

            if args_cml.scheduler == "plateau":
                lr_scheduler.step(total_val_loss)
            else:
                lr_scheduler.step()

            model.eval()

            all_preds = np.zeros((len(test_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT))
            all_reals = np.zeros((len(test_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT))
            te_index_offset = 0

            with torch.no_grad():
                for batch in test_dataloader:
                    batch = [comp.to(accelerator.device) for comp in batch]

                    pred = model(batch)
                    real = batch[0].y

                    all_preds[te_index_offset:te_index_offset + len(pred)] = pred.cpu().detach().numpy()
                    all_reals[te_index_offset:te_index_offset + len(real)] = real.cpu().detach().numpy()

                    te_index_offset += arg.NODE_COUNT * arg.BATCH_SIZE

            test_rmse = rmse(np.array(all_preds), np.array(all_reals))
            model.train()
            accelerator.print("RMSE on Testing Set", test_rmse)
            TEST_RMSE.append(test_rmse)

        accelerator.wait_for_everyone()
        accelerator.print("*" * 30)
        accelerator.print("Finish Training! Start Final Testing Evaluation!")
        accelerator.print("*" * 30)

        best_model = build_model(args_cml, arg, accelerator)
        best_model.load_state_dict(torch.load(os.path.join(arg.EXPERIMENT_NAME, f"model_{SPLIT}.pt"), map_location=accelerator.device))
        best_model.eval()

        all_preds = np.zeros((len(test_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT))
        all_reals = np.zeros((len(test_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT))
        te_index_offset = 0

        with torch.no_grad():
            for batch in test_dataloader:
                batch = [comp.to(accelerator.device) for comp in batch]

                pred = best_model(batch)
                real = batch[0].y

                all_preds[te_index_offset:te_index_offset + len(pred)] = pred.cpu().detach().numpy()
                all_reals[te_index_offset:te_index_offset + len(real)] = real.cpu().detach().numpy()

                te_index_offset += arg.NODE_COUNT * arg.BATCH_SIZE

        rmse_pred = rmse(np.array(all_preds), np.array(all_reals))

        accelerator.print("===================================")
        accelerator.print("Experiment: ", arg.EXPERIMENT_NAME)
        accelerator.print("RMSE: ", rmse_pred)
        accelerator.print("Pred Mean: ", np.mean(all_preds))
        accelerator.print("Pred Std: ", np.std(all_preds))
        accelerator.print("Real Mean: ", np.mean(all_reals))
        accelerator.print("Real Std: ", np.std(all_reals))
        accelerator.print("Dim RMSE: ", rmse_dim(np.array(all_preds), np.array(all_reals), 0))

        np.save(os.path.join(arg.EXPERIMENT_NAME, f"preds{SPLIT}.npy"), all_preds)
        np.save(os.path.join(arg.EXPERIMENT_NAME, f"reals{SPLIT}.npy"), all_reals)
        np.save(os.path.join(arg.EXPERIMENT_NAME, f"losses{SPLIT}.npy"), np.array([TRAINING_LOSSES, VALIDATION_LOSSES]))
        np.save(os.path.join(arg.EXPERIMENT_NAME, f"rmse{SPLIT}.npy"), np.array([TRAINING_RMSE, VALIDATION_RMSE, TEST_RMSE]))


if __name__ == "__main__":
    main()
