"""Main training script for the standard K-STEMIT shallow-to-deep experiments."""

import torch
import argparse
from types import SimpleNamespace
import os
from utils import split_dataset, load_dill, rmse, rmse_dim, PolyLR, save_model
from model_mb import IceSheetModel_Multibranch1, IceSheetModel_Multibranch1_NoClamp
from model_mb import IceSheetModel_Multibranch_ablation1, IceSheetModel_Multibranch_ablation1_NonAdaptive, IceSheetModel_Multibranch_ablation2, IceSheetModel_Multibranch_ablation3, IceSheetModel_Multibranch_ablation4, IceSheetModel_Multibranch_ablation5, IceSheetModel_Multibranch_ablation6
from model_fused import IceSheetModel_GCN, IceSheetModel_GAT, IceSheetModel_SAGE
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ExponentialLR, StepLR

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DataLoaderConfiguration
from torch_geometric import seed_everything

def parse_args() -> argparse.Namespace:
    '''
    Get Command Line Arguments
    '''

    parser = argparse.ArgumentParser(description='Internal Ice Layer')   
    parser.add_argument('--model', type=str, default='GCN', help='Model Name')
    # parser.add_argument('--dataset', type=str, default='OldDataset', help='Dataset Name')
    parser.add_argument('--batch', type=int, default=1, help='Batch Size')
    parser.add_argument('--epoch', type=int, default=300, help='Epoch')
    parser.add_argument('--adaptive', default="False", help='Use Adaptive')
    parser.add_argument('--ablation', default="False", help='Ablation')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
    parser.add_argument('--featureablation', default='0000000', help="Feature Ablation")
    parser.add_argument('--folder', default='Experiment', help="Folder Name For Saving")
    parser.add_argument('--scheduler', default='poly', help='Learning Rate Scheduler')
    parser.add_argument('--schedulerargs', type=float, help='Learning Rate Scheduler Argument')
    parser.add_argument('--eta_min', type=float, default=1e-7, help='Minimum Learning Rate for Cosine Annealing')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight Decay for Optimizer')

    args = parser.parse_args()
    return args


def main():
    ###########
    #Set up the parameters
    ###########
    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)
    set_seed(seed)

    args_cml = parse_args() # Load CML arguments
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(device_placement=True, step_scheduler_with_optimizer= False, dataloader_config = dataloader_config)
    accelerator.print("Args:", args_cml)
    device = accelerator.device

    #Create experiment arguments
    arg = SimpleNamespace()
    arg.FEATURE_ABLATION = args_cml.featureablation 
    arg.REMOVE_ALL_PHYSICAL_PARAMS = True
    arg.NODE_COUNT = 256 # Node counts in our graph
    arg.FEATURE_COUNT = 10 # Latitude, longitude, thickness and 7 physics parameters
    # The name of the folder to save results to.
    arg.EXPERIMENT_NAME = args_cml.folder 
    # How many layer thicknesses to predict.
    arg.LAYER_PREDICT_COUNT = 15
    # How many "feature layers" to use.
    arg.LAYER_FEATURE_COUNT = 5
    # Whether to predict deep-to-shallow (False) or shallow-to-deep (True)
    arg.PREDICT_HISTORIC = False

    # Whether or not to add an adaptive layer to the GCN-LSTM or GraphSAGE-LSTM.
    arg.ADAPTIVE = args_cml.adaptive.lower() == 'true'

    # Which model to use.
    arg.USE_GAT = False
    arg.USE_SAGE = False
    arg.USE_Multi1 = False
    arg.USE_Multi1_NoClamp = False
    arg.USE_Ablation1 = False
    arg.USE_Ablation1_NonAdaptive = False
    arg.USE_Ablation2 = False
    arg.USE_Ablation3 = False
    arg.USE_Ablation4 = False
    arg.USE_Ablation5 = False
    arg.USE_Ablation6 = False

    #arg.Model_Name = args_cml.model + "_" + str(args_cml.batch)
    if args_cml.model == "GAT":
        arg.USE_GAT = True
        accelerator.print("USE GAT LSTM")

    elif args_cml.model == "SAGE":
        arg.USE_SAGE = True
        accelerator.print("USE SAGE LSTM")

    elif args_cml.model == 'Multi1':
        arg.USE_Multi1 = True
        accelerator.print("USE Multi Branch 1")

    elif args_cml.model == 'Multi1NoClamp':
        arg.USE_Multi1_NoClamp = True
        accelerator.print("USE Multi Branch 1 No Clamp")

    elif args_cml.model == 'Ablation1':
        arg.USE_Ablation1 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 1")

    elif args_cml.model == 'Ablation1NonAdaptive':
        arg.USE_Ablation1_NonAdaptive = True
        accelerator.print("USE Multi Branch 1 Ablation Study 1 Non-Adaptive")

    elif args_cml.model == 'Ablation2':
        arg.USE_Ablation2 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 2")

    elif args_cml.model == 'Ablation3':
        arg.USE_Ablation3 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 3")

    elif args_cml.model == 'Ablation4':
        arg.USE_Ablation4 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 4")

    elif args_cml.model == 'Ablation5':
        arg.USE_Ablation5 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 5")

    elif args_cml.model == 'Ablation6':
        arg.USE_Ablation6 = True
        accelerator.print("USE Multi Branch 1 Ablation Study 6")

    # The number of channels within each of the three linear layers in the model.
    arg.DIMENSIONALITIES = [ 256, 128, 64 ]

    # The initial learning rate for the model.
    arg.INITIAL_LEARNING_RATE = args_cml.lr
    arg.eta_min = args_cml.eta_min
    arg.weight_decay = args_cml.weight_decay

    # Whether or not to run a physical parameter ablation study.
    # In the output experiment name, each feature is shown as either a 1 (included) or 0 (excluded).
    # The order of the features is as follows:
    # 1: Snow mass balance
    # 2: Average yearly surface temp
    # 3: Height change due to refreezing
    # 4: Height change due to melt
    # 5: Amount of snow pack
    # 6: Snow density
    # 7: Elevation
    # As an example, an experiment named GAT_TEST_100001 means that snow mass balance and elevation were used but no others.
    #args.RUN_ABLATION_STUDY = args_cml.ablation #False
    #print(args_cml.ablation.lower(), type(args_cml.adaptive.lower()))
    arg.RUN_ABLATION_STUDY = args_cml.ablation.lower() == 'true'
    # The name of the experiment that generated the desired dataset.
    #arg.DATASET_EXPERIMENT = arg.EXPERIMENT_NAME
    # Batch Size:
    arg.BATCH_SIZE = args_cml.batch
    # Epoch
    arg.epoch = args_cml.epoch
    ############ END HERE !

    arg.REMOVE_ALL_PHYSICAL_PARAMS = not arg.RUN_ABLATION_STUDY
    if arg.REMOVE_ALL_PHYSICAL_PARAMS:
        accelerator.print("Change Feature Counts")
        arg.FEATURE_COUNT = 3

    if not os.path.exists(arg.EXPERIMENT_NAME + "/"):
        os.mkdir(arg.EXPERIMENT_NAME + "/")

    ############################
    #Step 1: Loading Whole Dataset
    ############################
    dataset = load_dill('/home/zel220/Internal_Ice_Layer_GNN_Dataset/Shallow-To-Deep/mixture/Shallow-To-Deep_Mix_WithPhysics/dataset')
    arg.SPLIT_FILE = '/home/zel220/Internal_Ice_Layer_GNN_Dataset/Shallow-To-Deep/mixture/Shallow-To-Deep_Mix_WithPhysics/splits.npy'

    ############################
    #Step 2: Training
    ############################
    for i in range(5): 

        SPLIT = i

        # Get dataset and split it into training, validation, and testing sets
        accelerator.print("Splitting Dataset")
        accelerator.print("Split Number: ", SPLIT)
        training_dataset, val_dataset, testing_dataset = split_dataset(dataset, SPLIT, arg.SPLIT_FILE)
        accelerator.print("Training: ", len(training_dataset))
        accelerator.print("Validation: ", len(val_dataset))
        accelerator.print("Testing: ", len(testing_dataset))

        # Normalize Training, Validation, and Testing Datasets
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
        # Normalize Training Dataset
        for graph_collection in training_dataset:
            for graph in graph_collection:
                graph.x -= train_features_mean
                graph.x /= train_features_std
        # Normalize Validation Dataset
        for graph_collection in val_dataset:
            for graph in graph_collection:
                graph.x -= train_features_mean
                graph.x /= train_features_std
        # Normalize Testing Dataset
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
                            gc[i].x[:, 3+j] = 0

            for gc in val_dataset:
                for i in range(len(gc)):
                    for j in range(len(arg.FEATURE_ABLATION)):
                        if arg.FEATURE_ABLATION[j] == "0":
                            gc[i].x[:, 3+j] = 0

            for gc in testing_dataset:
                for i in range(len(gc)):
                    for j in range(len(arg.FEATURE_ABLATION)):
                        if arg.FEATURE_ABLATION[j] == "0":
                            gc[i].x[:, 3+j] = 0

        #Generate Data Loader
        train_dataloader = DataLoader(training_dataset, shuffle=False, batch_size=arg.BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=arg.BATCH_SIZE)
        test_dataloader = DataLoader(testing_dataset, shuffle=False, batch_size=arg.BATCH_SIZE)
        accelerator.print("Train Loader", len(train_dataloader))
        accelerator.print("Val Loader", len(val_dataloader))
        accelerator.print("Test Loader", len(test_dataloader))
        
        #Create Model
        if arg.USE_GAT:
            model = IceSheetModel_GAT(arg).to(accelerator.device)
        elif arg.USE_SAGE:
            model = IceSheetModel_SAGE(arg).to(accelerator.device)
        elif arg.USE_Multi1:
            model = IceSheetModel_Multibranch1(arg).to(accelerator.device)
        elif arg.USE_Multi1_NoClamp:
            model = IceSheetModel_Multibranch1_NoClamp(arg).to(accelerator.device)
        elif arg.USE_Ablation1:
            model = IceSheetModel_Multibranch_ablation1(arg).to(accelerator.device)
        elif arg.USE_Ablation1_NonAdaptive:
            model = IceSheetModel_Multibranch_ablation1_NonAdaptive(arg).to(accelerator.device)
        elif arg.USE_Ablation2:
            model = IceSheetModel_Multibranch_ablation2(arg).to(accelerator.device)
        elif arg.USE_Ablation3:
            model = IceSheetModel_Multibranch_ablation3(arg).to(accelerator.device)
        elif arg.USE_Ablation4:
            model = IceSheetModel_Multibranch_ablation4(arg).to(accelerator.device)
        elif arg.USE_Ablation5:
            model = IceSheetModel_Multibranch_ablation5(arg).to(accelerator.device)
        elif arg.USE_Ablation6:
            model = IceSheetModel_Multibranch_ablation6(arg).to(accelerator.device)
        else:
            accelerator.print("Use Model GCN LSTM")
            model = IceSheetModel_GCN(arg).to(accelerator.device)

        #Create Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.INITIAL_LEARNING_RATE, weight_decay=arg.weight_decay)
        if args_cml.scheduler == 'poly':
            lr_scheduler = PolyLR(optimizer, max_iters=arg.epoch, power=args_cml.schedulerargs)
        elif args_cml.scheduler == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=int(args_cml.schedulerargs), eta_min=arg.eta_min)
        elif args_cml.scheduler == 'warmup':
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(args_cml.schedulerargs), eta_min=1e-6)
        elif args_cml.scheduler == 'exp':
            lr_scheduler = ExponentialLR(optimizer, gamma=0.99)
        elif args_cml.scheduler == 'step':
            lr_scheduler = StepLR(optimizer, step_size=int(args_cml.schedulerargs), gamma=0.5) #
        elif args_cml.scheduler == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=int(args_cml.schedulerargs), threshold=1e-6, cooldown=5, verbose=True, min_lr=1e-6)
        
        # Prepare everything with the accelerator
        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader)

        #Everything is ready, start training
        accelerator.print("Everything is Ready! Let's train our model!")

        TRAINING_LOSSES = []
        VALIDATION_LOSSES = []
        TRAINING_EPOCHS = 0

        TRAINING_RMSE = []
        VALIDATION_RMSE = []
        TEST_RMSE = []

        best_val_loss = 999999
        best_model = None
        best_model_epoch = 0
        seed_everything(1337)
        set_seed(1337)

        for epoch in tqdm(range(arg.epoch), disable=not accelerator.is_local_main_process):
            tr_all_preds = np.zeros(( len(train_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT ))
            tr_all_reals = np.zeros(( len(train_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT ))

            tr_index_offset = 0
            total_loss = 0
            
            #Train loop
            model.train()
            for batch in train_dataloader:
                batch = [comp.to(accelerator.device) for comp in batch]
                optimizer.zero_grad()

                pred = model(batch)
                real = batch[0].y 

                loss = F.mse_loss(pred.float(), real.float())
                total_loss += loss.item()

                tr_all_preds[tr_index_offset : tr_index_offset + len(pred)] = pred.cpu().detach().numpy()
                tr_all_reals[tr_index_offset : tr_index_offset + len(real)] = real.cpu().detach().numpy()

                accelerator.backward(loss) #.backward()
                optimizer.step()
                
                tr_index_offset += arg.NODE_COUNT * arg.BATCH_SIZE

            accelerator.wait_for_everyone()

            # Calculate Training RMSE and Loss
            train_rmse = rmse(np.array(tr_all_preds), np.array(tr_all_reals))
            total_loss = total_loss/len(train_dataloader)
            accelerator.print(f"Epoch #{epoch} Device #{device} Training RMSE: ", train_rmse, "; Loss: ", total_loss)
            TRAINING_LOSSES.append(total_loss)
            TRAINING_RMSE.append(train_rmse)

            # Validation on validation set
            val_all_preds = np.zeros(( len(val_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT ))
            val_all_reals = np.zeros(( len(val_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT ))
            model.eval()

            with torch.no_grad():
                val_loss = 0
                total_val_loss = 0
                val_index_offset = 0

                for batch in val_dataloader:
                    batch = [comp.to(accelerator.device) for comp in batch]
                    pred = model(batch)
                    real = batch[0].y 

                    val_loss = F.mse_loss(pred.float(), real.float())
                    total_val_loss += val_loss

                    val_all_preds[val_index_offset : val_index_offset+len(pred)] = pred.cpu().detach().numpy()
                    val_all_reals[val_index_offset : val_index_offset+len(real)] = real.cpu().detach().numpy()

                    val_index_offset += arg.NODE_COUNT * arg.BATCH_SIZE

            accelerator.wait_for_everyone()
            
            # Calculate Validation RMSE and Loss
            model.train()
            val_rmse = rmse(np.array(val_all_preds), np.array(val_all_reals))
            total_val_loss = total_val_loss/len(val_dataloader)
            accelerator.print(f"Epoch #{epoch} Validation RMSE: ", val_rmse, "; Loss: ", total_val_loss.item())
            VALIDATION_RMSE.append(val_rmse)
            VALIDATION_LOSSES.append(total_val_loss.item())
            TRAINING_EPOCHS += 1

            # Save Best Results
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_model = model.state_dict()
                best_model_epoch = epoch
                save_model(model, arg, SPLIT, accelerator)
                accelerator.print("Best model! Val loss = ", best_val_loss)
            accelerator.print("Best epoch so far: ", best_model_epoch)

            # Update Learning Rate
            if args_cml.scheduler == 'plateau':
                lr_scheduler.step(total_val_loss)
            else:
                lr_scheduler.step()
                

            # Test on testing dataset
            model.eval()

            all_preds = np.zeros(( len(test_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT ))     
            all_reals = np.zeros(( len(test_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT ))
            te_index_offset = 0

            with torch.no_grad():
                for batch in test_dataloader:
                    batch = [comp.to(accelerator.device) for comp in batch]

                    pred = model(batch)
                    real = batch[0].y 

                    all_preds[te_index_offset : te_index_offset + len(pred)] = pred.cpu().detach().numpy()
                    all_reals[te_index_offset : te_index_offset + len(real)] = real.cpu().detach().numpy()

                    te_index_offset += arg.NODE_COUNT * arg.BATCH_SIZE

            test_rmse = rmse(np.array(all_preds), np.array(all_reals))
            model.train()
            accelerator.print("RMSE on Testing Set", test_rmse)
            TEST_RMSE.append(test_rmse)

        accelerator.wait_for_everyone()
        accelerator.print("*" * 30)
        accelerator.print("Finish Training! Start Final Testing Evaluation!")
        accelerator.print("*" * 30)

        # Load best model
        if arg.USE_GAT:
            best_model = IceSheetModel_GAT(arg).to(accelerator.device)
        elif arg.USE_SAGE:
            best_model = IceSheetModel_SAGE(arg).to(accelerator.device)
        elif arg.USE_Multi1:
            best_model = IceSheetModel_Multibranch1(arg).to(accelerator.device)
        elif arg.USE_Multi1_NoClamp:
            best_model = IceSheetModel_Multibranch1_NoClamp(arg).to(accelerator.device)
        elif arg.USE_Ablation1:
            best_model = IceSheetModel_Multibranch_ablation1(arg).to(accelerator.device)
        elif arg.USE_Ablation1_NonAdaptive:
            best_model = IceSheetModel_Multibranch_ablation1_NonAdaptive(arg).to(accelerator.device)
        elif arg.USE_Ablation2:
            best_model = IceSheetModel_Multibranch_ablation2(arg).to(accelerator.device)
        elif arg.USE_Ablation3:
            best_model = IceSheetModel_Multibranch_ablation3(arg).to(accelerator.device)
        elif arg.USE_Ablation4:
            best_model = IceSheetModel_Multibranch_ablation4(arg).to(accelerator.device)
        elif arg.USE_Ablation5:
            best_model = IceSheetModel_Multibranch_ablation5(arg).to(accelerator.device)
        elif arg.USE_Ablation6:
            best_model = IceSheetModel_Multibranch_ablation6(arg).to(accelerator.device)
        else:
            best_model = IceSheetModel_GCN(arg).to(accelerator.device)

        best_model.load_state_dict(torch.load(arg.EXPERIMENT_NAME +"/model_" + str(SPLIT) + ".pt"))
        best_model.eval()

        all_preds = np.zeros(( len(test_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT ))     

        all_reals = np.zeros(( len(test_dataloader) * arg.NODE_COUNT * arg.BATCH_SIZE, arg.LAYER_PREDICT_COUNT ))
        te_index_offset = 0

        with torch.no_grad():
            for batch in test_dataloader:
                batch = [comp.to(accelerator.device) for comp in batch]

                pred = best_model(batch)
                real = batch[0].y 

                all_preds[te_index_offset : te_index_offset + len(pred)] = pred.cpu().detach().numpy()
                all_reals[te_index_offset : te_index_offset + len(real)] = real.cpu().detach().numpy()

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
            
        np.save(arg.EXPERIMENT_NAME + "/preds" + str(SPLIT) + ".npy", all_preds)
        np.save(arg.EXPERIMENT_NAME + "/reals" + str(SPLIT) + ".npy", all_reals)
        np.save(arg.EXPERIMENT_NAME + "/losses" + str(SPLIT) + ".npy", np.array([TRAINING_LOSSES, VALIDATION_LOSSES]))
        np.save(arg.EXPERIMENT_NAME + "/rmse" + str(SPLIT) + ".npy", np.array([TRAINING_RMSE, VALIDATION_RMSE, TEST_RMSE]))

if __name__ == "__main__":
    main()
