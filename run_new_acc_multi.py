"""Training entry point for dataset_predict_xx multi-dataset K-STEMIT experiments."""

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

from model_fused import IceSheetModel_GAT
from model_fused import IceSheetModel_GCN
from model_fused import IceSheetModel_SAGE
from model_mb import IceSheetModel_Multibranch1
from model_mb import IceSheetModel_Multibranch_ablation1
from model_mb import IceSheetModel_Multibranch_ablation1_NonAdaptive
from model_mb import IceSheetModel_Multibranch_ablation2
from model_mb import IceSheetModel_Multibranch_ablation3
from model_mb import IceSheetModel_Multibranch_ablation4
from model_mb import IceSheetModel_Multibranch_ablation5
from model_mb import IceSheetModel_Multibranch_ablation6
from utils import PolyLR
from utils import ensure_splits
from utils import load_dill
from utils import rmse
from utils import rmse_dim
from utils import save_model
from utils import split_dataset


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = str(DEFAULT_WORKSPACE_ROOT / "Shallow-To-Deep-Multi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal Ice Layer Multi Dataset Training")
    parser.add_argument("--model", type=str, default="GCN", help="Model Name")
    parser.add_argument("--batch", type=int, default=1, help="Batch Size")
    parser.add_argument("--epoch", type=int, default=300, help="Epoch")
    parser.add_argument("--adaptive", default="False", help="Use Adaptive")
    parser.add_argument("--ablation", default="False", help="Ablation")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    parser.add_argument("--featureablation", default="0000000", help="Feature Ablation")
    parser.add_argument("--folder", default="Experiment", help="Folder Name For Saving")
    parser.add_argument("--scheduler", default="poly", help="Learning Rate Scheduler")
    parser.add_argument("--schedulerargs", type=float, help="Learning Rate Scheduler Argument")
    parser.add_argument("--eta_min", type=float, default=1e-7, help="Minimum Learning Rate for Cosine Annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight Decay for Optimizer")
    parser.add_argument("--predict-count", type=int, required=True, help="Predict count for dataset_predict_xx")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Root directory containing dataset_predict_xx files")
    parser.add_argument("--dataset-path", default=None, help="Optional explicit path to the dataset dill file")
    parser.add_argument("--split-file", default=None, help="Optional explicit split file path")
    parser.add_argument("--split-seed", type=int, default=1337, help="Seed used to create per-case split files")
    parser.add_argument("--split-count", type=int, default=5, help="Number of shuffled split permutations to save")
    parser.add_argument("--overwrite-splits", action="store_true", help="Regenerate split file even if a matching one exists")
    return parser.parse_args()


def resolve_dataset_path(args_cml: argparse.Namespace) -> tuple[str, str]:
    dataset_root = os.path.abspath(os.path.expanduser(args_cml.dataset_root))
    dataset_path = args_cml.dataset_path
    split_file = args_cml.split_file

    if dataset_path is None:
        dataset_path = os.path.join(dataset_root, f"dataset_predict_{args_cml.predict_count}")
    else:
        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

    if split_file is None:
        split_file = os.path.join(dataset_root, f"splits_predict_{args_cml.predict_count}.npy")
    else:
        split_file = os.path.abspath(os.path.expanduser(split_file))

    return dataset_path, split_file


def build_model(arg, accelerator):
    if arg.USE_GAT:
        return IceSheetModel_GAT(arg).to(accelerator.device)
    if arg.USE_SAGE:
        return IceSheetModel_SAGE(arg).to(accelerator.device)
    if arg.USE_Multi1:
        return IceSheetModel_Multibranch1(arg).to(accelerator.device)
    if arg.USE_Ablation1:
        return IceSheetModel_Multibranch_ablation1(arg).to(accelerator.device)
    if arg.USE_Ablation1_NonAdaptive:
        return IceSheetModel_Multibranch_ablation1_NonAdaptive(arg).to(accelerator.device)
    if arg.USE_Ablation2:
        return IceSheetModel_Multibranch_ablation2(arg).to(accelerator.device)
    if arg.USE_Ablation3:
        return IceSheetModel_Multibranch_ablation3(arg).to(accelerator.device)
    if arg.USE_Ablation4:
        return IceSheetModel_Multibranch_ablation4(arg).to(accelerator.device)
    if arg.USE_Ablation5:
        return IceSheetModel_Multibranch_ablation5(arg).to(accelerator.device)
    if arg.USE_Ablation6:
        return IceSheetModel_Multibranch_ablation6(arg).to(accelerator.device)
    return IceSheetModel_GCN(arg).to(accelerator.device)


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


def main():
    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)
    set_seed(seed)

    args_cml = parse_args()
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(device_placement=True, step_scheduler_with_optimizer=False, dataloader_config=dataloader_config)
    accelerator.print("Args:", args_cml)
    device = accelerator.device

    dataset_path, split_file = resolve_dataset_path(args_cml)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_dill(dataset_path)
    dataset_feature_count = None
    if len(dataset) > 0 and len(dataset[0]) > 0:
        dataset_feature_count = dataset[0][0].x.shape[1]

    expected_feature_count = 10 if args_cml.ablation.lower() == "true" else 3
    if dataset_feature_count is not None and dataset_feature_count != expected_feature_count:
        raise ValueError(
            "Dataset feature count mismatch for "
            f"{dataset_path}: found {dataset_feature_count}, expected {expected_feature_count}. "
            "Rebuild the multi dataset with the matching physical-feature setting before training."
        )

    split_file, generated_split_file = ensure_splits(
        len(dataset),
        split_file,
        seed=args_cml.split_seed,
        split_count=args_cml.split_count,
        overwrite=args_cml.overwrite_splits,
    )

    accelerator.print("Dataset Path:", dataset_path)
    accelerator.print("Dataset Samples:", len(dataset))
    accelerator.print("Predict Count:", args_cml.predict_count)
    accelerator.print("Split File:", split_file)
    accelerator.print("Split File Status:", "generated" if generated_split_file else "reused")

    arg = SimpleNamespace()
    arg.FEATURE_ABLATION = args_cml.featureablation
    arg.REMOVE_ALL_PHYSICAL_PARAMS = True
    arg.NODE_COUNT = 256
    arg.FEATURE_COUNT = 10
    arg.EXPERIMENT_NAME = args_cml.folder
    arg.LAYER_PREDICT_COUNT = args_cml.predict_count
    arg.LAYER_FEATURE_COUNT = 5
    arg.PREDICT_HISTORIC = False
    arg.ADAPTIVE = args_cml.adaptive.lower() == "true"

    arg.USE_GAT = False
    arg.USE_SAGE = False
    arg.USE_Multi1 = False
    arg.USE_Ablation1 = False
    arg.USE_Ablation1_NonAdaptive = False
    arg.USE_Ablation2 = False
    arg.USE_Ablation3 = False
    arg.USE_Ablation4 = False
    arg.USE_Ablation5 = False
    arg.USE_Ablation6 = False

    if args_cml.model == "GAT":
        arg.USE_GAT = True
        accelerator.print("USE GAT LSTM")
    elif args_cml.model == "SAGE":
        arg.USE_SAGE = True
        accelerator.print("USE SAGE LSTM")
    elif args_cml.model == "Multi1":
        arg.USE_Multi1 = True
        accelerator.print("USE Multi Branch 1")
    elif args_cml.model == "Ablation1":
        arg.USE_Ablation1 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 1")
    elif args_cml.model == "Ablation1NonAdaptive":
        arg.USE_Ablation1_NonAdaptive = True
        accelerator.print("USE Multi Branch 1 Ablation Study 1 Non-Adaptive")
    elif args_cml.model == "Ablation2":
        arg.USE_Ablation2 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 2")
    elif args_cml.model == "Ablation3":
        arg.USE_Ablation3 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 3")
    elif args_cml.model == "Ablation4":
        arg.USE_Ablation4 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 4")
    elif args_cml.model == "Ablation5":
        arg.USE_Ablation5 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 5")
    elif args_cml.model == "Ablation6":
        arg.USE_Ablation6 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 6")

    arg.DIMENSIONALITIES = [256, 128, 64]
    arg.INITIAL_LEARNING_RATE = args_cml.lr
    arg.eta_min = args_cml.eta_min
    arg.weight_decay = args_cml.weight_decay
    arg.RUN_ABLATION_STUDY = args_cml.ablation.lower() == "true"
    arg.BATCH_SIZE = args_cml.batch
    arg.epoch = args_cml.epoch
    arg.SPLIT_FILE = split_file

    arg.REMOVE_ALL_PHYSICAL_PARAMS = not arg.RUN_ABLATION_STUDY
    if arg.REMOVE_ALL_PHYSICAL_PARAMS:
        accelerator.print("Change Feature Counts")
        arg.FEATURE_COUNT = 3

    os.makedirs(arg.EXPERIMENT_NAME, exist_ok=True)

    for split_idx in range(args_cml.split_count):
        SPLIT = split_idx

        accelerator.print("Splitting Dataset")
        accelerator.print("Split Number:", SPLIT)
        training_dataset, val_dataset, testing_dataset = split_dataset(dataset, SPLIT, arg.SPLIT_FILE)
        accelerator.print("Training:", len(training_dataset))
        accelerator.print("Validation:", len(val_dataset))
        accelerator.print("Testing:", len(testing_dataset))

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
        accelerator.print("Normalized means and stds:")
        accelerator.print(train_features_mean)
        accelerator.print(train_features_std)

        for graph_collection in training_dataset:
            for graph in graph_collection:
                graph.x -= train_features_mean
                graph.x /= train_features_std

        for graph_collection in val_dataset:
            for graph in graph_collection:
                graph.x -= train_features_mean
                graph.x /= train_features_std

        for graph_collection in testing_dataset:
            for graph in graph_collection:
                graph.x -= train_features_mean
                graph.x /= train_features_std

        if not arg.REMOVE_ALL_PHYSICAL_PARAMS:
            accelerator.print("Training with Physical Features!!!!")
            accelerator.print("Removing Unnecessary Physical Features")
            for gc in training_dataset:
                for i in range(len(gc)):
                    for j in range(len(arg.FEATURE_ABLATION)):
                        if arg.FEATURE_ABLATION[j] == "0":
                            gc[i].x[:, 3 + j] = 0

            for gc in val_dataset:
                for i in range(len(gc)):
                    for j in range(len(arg.FEATURE_ABLATION)):
                        if arg.FEATURE_ABLATION[j] == "0":
                            gc[i].x[:, 3 + j] = 0

            for gc in testing_dataset:
                for i in range(len(gc)):
                    for j in range(len(arg.FEATURE_ABLATION)):
                        if arg.FEATURE_ABLATION[j] == "0":
                            gc[i].x[:, 3 + j] = 0

        train_dataloader = DataLoader(training_dataset, shuffle=False, batch_size=arg.BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=arg.BATCH_SIZE)
        test_dataloader = DataLoader(testing_dataset, shuffle=False, batch_size=arg.BATCH_SIZE)
        accelerator.print("Train Loader", len(train_dataloader))
        accelerator.print("Val Loader", len(val_dataloader))
        accelerator.print("Test Loader", len(test_dataloader))

        model = build_model(arg, accelerator)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.INITIAL_LEARNING_RATE, weight_decay=arg.weight_decay)
        lr_scheduler = build_scheduler(args_cml, optimizer, arg)

        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader)

        accelerator.print("Everything is Ready! Let's train our model!")

        TRAINING_LOSSES = []
        VALIDATION_LOSSES = []
        TRAINING_EPOCHS = 0

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
            TRAINING_EPOCHS += 1

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

        best_model = build_model(arg, accelerator)
        best_model.load_state_dict(torch.load(os.path.join(arg.EXPERIMENT_NAME, f"model_{SPLIT}.pt")))
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
