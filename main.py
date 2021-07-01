# This is a sample Python script.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from congestion_predict.data_sets import load_datasets
from congestion_predict.models import load_model
import congestion_predict.evaluation as eval_util
import time
import json


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def train_model(model_name, model, train_set, valid_set, mean_torch, stddev_torch):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    model.to(device)  # Put the network on GPU if present

    criterion = nn.MSELoss()  # L2 Norm
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # ADAM with lr=10^-4  // Changed lr=1e-3 --> lr=1e-4
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)  # exponential decay every epoch = 2000iter  // Changed comment

    # sys.stdout = open("log.txt", "w")
    torch.cuda.empty_cache()
    min_loss = 1000000
    loss_plot_train = []
    mse_plot_valid = []
    mae_plot_valid = []

    if model_name == 'eRCNNSeqIter':
        weight = 0
        weight_add = ((100 / epochs) / (len(train_set) / batch_size)) * 100

    for epoch in range(epochs):  # 10 epochs
        print(f"******************Epoch {epoch}*******************\n\n")
        model.train()
        torch.autograd.set_detect_anomaly(True)
        losses_train = []

        # Train
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.permute(1, 0, 2, 3, 4)
            targets = targets.permute(1, 0, 2)

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Zero the gradients
            loss = torch.zeros(1, requires_grad=True)

            if model_name == 'eRCNNSeqIter':
                outputs = model(inputs.float(), targets.float(), weight)  # // Changed float
            else:
                outputs = model(inputs.float(), targets.float())  # // Changed float

            if target_norm:
                if model_name == 'eRCNNSeqLin':
                    loss = criterion(outputs * stddev_torch + mean_torch, targets[-1] * stddev_torch + mean_torch)
                else:
                    loss = criterion(outputs * stddev_torch + mean_torch,
                                     targets[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
            else:
                if model_name == 'eRCNNSeqLin':
                    loss = criterion(outputs, targets[-1])
                else:
                    loss = criterion(outputs, targets[-out_seq:].permute(1, 0, 2))

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 40)  # gradient clipping norm // Changed comment
            optimizer.step()  # Updated the weights
            losses_train.append(loss.item())
            end = time.time()

            if (batch_idx + 1) % batch_div == 0:
                # print(losses_train)
                print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (
                batch_idx, np.mean(losses_train), end - start))
                loss_plot_train.append(losses_train)
                losses_train = []
                if model_name == 'eRCNNSeqIter':
                    weight += weight_add
                start = time.time()
        #scheduler.step()  # // Changed comment

        # Evaluate
        model.eval()
        mse_valid = []
        mae_valid = []
        with torch.no_grad():
            for batch_idx, (inputs_valid, targets_valid) in enumerate(valid_loader):
                inputs_valid = inputs_valid.permute(1, 0, 2, 3, 4)
                targets_valid = targets_valid.permute(1, 0, 2)
                inputs_valid, targets_valid = inputs_valid.to(device), targets_valid.to(device)

                outputs_valid = model(inputs_valid.float(), targets_valid.float())  # // Changed float
                if (target_norm):
                    if model_name == 'eRCNNSeqLin':
                        loss_mse = criterion(outputs_valid * stddev_torch + mean_torch,
                                             targets_valid[-1] * stddev_torch + mean_torch)
                        loss_mae = criterion2(outputs_valid * stddev_torch + mean_torch,
                                              targets_valid[-1] * stddev_torch + mean_torch)
                    else:
                        loss_mse = criterion(outputs_valid * stddev_torch + mean_torch,
                                             targets_valid[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
                        loss_mae = criterion2(outputs_valid * stddev_torch + mean_torch,
                                              targets_valid[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
                else:
                    if model_name == 'eRCNNSeqLin':
                        loss_mse = criterion(outputs_valid, targets_valid[-1])
                        loss_mae = criterion2(outputs_valid, targets_valid[-1])
                    else:
                        loss_mse = criterion(outputs_valid, targets_valid[-out_seq:].permute(1, 0, 2))
                        loss_mae = criterion2(outputs_valid, targets_valid[-out_seq:].permute(1, 0, 2))
                mse_valid.append(loss_mse.item())
                mae_valid.append(loss_mae.item())
                if (batch_idx + 1) % batch_div == 0:
                    print('Batch Index : %d MSE : %.3f' % (batch_idx, np.mean(mse_valid)))
                    print('Batch Index : %d MAE : %.3f' % (batch_idx, np.mean(mae_valid)))
                    mse_plot_valid.append(mse_valid)
                    mae_plot_valid.append(mae_valid)
                    mse_valid = []
                    mae_valid = []
        print('--------------------------------------------------------------')
        print(f"Final Training LOSS in Epoch {epoch} = {np.mean(loss_plot_train[-10:])}")
        print(f"Final Validation MSE in Epoch {epoch} = {np.mean(mse_plot_valid[-10:])}")
        print(f"Final Validation MAE in Epoch {epoch} = {np.mean(mae_plot_valid[-10:])}\n")

        if np.mean(mse_plot_valid[-10:]) < min_loss:
            min_loss = np.mean(mse_plot_valid[-10:])
            no_better = 0
            print('Saving best model\n')
            torch.save(model.state_dict(), result_folder + 'best_observer.pt')
        else:
            no_better += 1
            if no_better >= patience:
                print('Finishing by Early Stopping')
                break

    with open(result_folder + f'loss_plot_train_{target}.txt', 'w') as filehandle:
        json.dump(loss_plot_train, filehandle)
    with open(result_folder + f'mse_plot_valid_{target}.txt', 'w') as filehandle:
        json.dump(mse_plot_valid, filehandle)
    with open(result_folder + f'mae_plot_valid_{target}.txt', 'w') as filehandle:
        json.dump(mae_plot_valid, filehandle)

    return loss_plot_train, mse_plot_valid, mae_plot_valid


def test_model(model_name, model, model_file_name, test_set):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()  # L2 Norm
    criterion2 = nn.L1Loss()

    # %% Testing
    print(f"****************** Testing *******************\n\n")
    model.load_state_dict(torch.load(result_folder + model_file_name, map_location=torch.device(device)))
    model.eval()
    mse_plot_test = []
    mae_plot_test = []
    mse_test = []
    mae_test = []
    with torch.no_grad():
        for batch_idx, (inputs_test, targets_test) in enumerate(test_loader):
            inputs_test = inputs_test.permute(1, 0, 2, 3, 4)
            targets_test = targets_test.permute(1, 0, 2)
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

            outputs_test = model(inputs_test.float(), targets_test.float())  # // Changed float
            if (target_norm):
                if model_name == 'eRCNNSeqLin':
                    loss_mse = criterion(outputs_test * stddev_torch + mean_torch,
                                         targets_test[-1] * stddev_torch + mean_torch)
                    loss_mae = criterion2(outputs_test * stddev_torch + mean_torch,
                                          targets_test[-1] * stddev_torch + mean_torch)
                else:
                    loss_mse = criterion(outputs_test * stddev_torch + mean_torch,
                                         targets_test[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
                    loss_mae = criterion2(outputs_test * stddev_torch + mean_torch,
                                          targets_test[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
            else:
                if model_name == 'eRCNNSeqLin':
                    loss_mse = criterion(outputs_test, targets_test[-1])
                    loss_mae = criterion2(outputs_test, targets_test[-1])
                else:
                    loss_mse = criterion(outputs_test, targets_test[-out_seq:].permute(1, 0, 2))
                    loss_mae = criterion2(outputs_test, targets_test[-out_seq:].permute(1, 0, 2))
            mse_test.append(loss_mse.item())
            mae_test.append(loss_mae.item())
            if (batch_idx + 1) % batch_div == 0:
                print('Batch Index : %d MSE : %.3f' % (batch_idx, np.mean(mse_test)))
                print('Batch Index : %d MAE : %.3f' % (batch_idx, np.mean(mae_test)))
                mse_plot_test.append(mse_test)
                mae_plot_test.append(mae_test)
                mse_test = []
                mae_test = []

    with open(result_folder + f'mse_plot_test_{target}.txt', 'w') as filehandle:
        json.dump(mse_plot_test, filehandle)
    with open(result_folder + f'mae_plot_test_{target}.txt', 'w') as filehandle:
        json.dump(mae_plot_test, filehandle)

    return mse_plot_test, mae_plot_test


def print_results(result_folder, target, loss_plot_train, mse_plot_valid, mae_plot_valid, mse_plot_test, mae_plot_test):
    # %% Final Results
    with open(result_folder + f'final_results_{target}.txt', 'w') as filehandle:
        filehandle.write(f"Final Training LOSS = {np.mean(loss_plot_train[-10:])}\n\n")
        filehandle.write(f"Final Validation MSE = {np.mean(mse_plot_valid[-10:])}\n")
        filehandle.write(f"Final Validation MAE = {np.mean(mae_plot_valid[-10:])}\n\n")
        filehandle.write(f"Testing MSE = {np.mean(mse_plot_test)}\n")
        filehandle.write(f"Testing MAE = {np.mean(mae_plot_test)}")

    print(f"Final Training LOSS = {np.mean(loss_plot_train[-10:])}")
    print(f"Final Validation MSE = {np.mean(mse_plot_valid[-10:])}")
    print(f"Final Validation MAE = {np.mean(mae_plot_valid[-10:])}")
    print(f"Testing MSE = {np.mean(mse_plot_test)}")
    print(f"Testing MAE = {np.mean(mae_plot_test)}")

    # %% Plotting
    eval_util.plot_loss('MSE', f'loss_plot_train_{target}.txt', folder=result_folder)
    eval_util.plot_mse(f'mse_plot_valid_{target}.txt', target, folder=result_folder)
    eval_util.plot_mae(f'mae_plot_valid_{target}.txt', target, folder=result_folder)
    eval_util.plot_mse(f'mse_plot_test_{target}.txt', target, folder=result_folder)
    eval_util.plot_mae(f'mae_plot_test_{target}.txt', target, folder=result_folder)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_idx = input('Select a dataset:\n'
                         '1. California I5\n'
                         '2. METR_LA\n'
                         '3. Las Vegas I15\n')

    pred_variable = input('Select a prediction variable:\n'
                          '1. Flow\n'
                          '2. Occupancy\n'
                          '3. Speed\n')

    model_idx = input('Select a model:\n'
                       '1. eRCNNSeqIter\n'
                       '2. eRCNNSeqLin\n'
                       '3. eREncDec\n')

    # Dataset Variables
    datasets_list = ['cali_i5', 'metr_la', 'vegas_i15']
    dataset = datasets_list[int(dataset_idx) - 1]

    # Data Variables
    target = int(pred_variable) - 1
    variables_list = ['flow', 'occupancy', 'speed']
    pred_window = 12
    out_seq = pred_window  # Size of the out sequence
    pred_type = 'solo'
    seq_size = 12  # Best 12 for eREncDec
    image_size = 72  # for the cali_i5 dataset
    target_norm = False
    batch_div = 40  # 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
    # device = "cpu"
    torch.manual_seed(50)  # all exactly the same (model parameters initialization and data split)

    # Model Variables
    models_list = ['eRCNNSeqIter', 'eRCNNSeqLin', 'eREncDec']
    model_name = models_list[int(model_idx) - 1]
    pred_detector_list = ['all_iter', 'all_lin', 'all_iter']
    pred_detector = pred_detector_list[int(model_idx) - 1]
    result_folder_list = ['resultados/eRCNN/eRCNNSeqIter/',
                          'resultados/eRCNN/eRCNNSeqLin/',
                          'resultados/EncoderDecoder/Results/']
    result_folder = result_folder_list[int(model_idx) - 1]
    # eREncDec Model Unique Variables
    hidden_size_rec = 40  # 7/20/40/50/70 --> best 50
    num_layers_rec = 2  # 2/3/4
    seqlen_rec = 6  # 8/12

    # Training Variables
    if model_name == 'eRCNNSeqIter':
        epochs = 10
    else:
        epochs = 10000
    batch_size = 50  # Training Batch size
    patience = 5

    print(f'Starting: Dataset: {dataset}, Model: {model_name}, Target: {variables_list[target]}')

    train_set, valid_set, test_set, mean_torch, stddev_torch = load_datasets(dataset,
                                                                             pred_type,
                                                                             pred_window,
                                                                             pred_detector,
                                                                             target,
                                                                             seq_size,
                                                                             image_size,
                                                                             target_norm,
                                                                             device=device)
    print(f"Size of train_set = {len(train_set)}")
    print(f"Size of valid_set = {len(valid_set)}")
    print(f"Size of test_set = {len(test_set)}")

    # %% View Image sample
    image_seq, label = train_set[0]
    print(image_seq)
    print(label)

    model = load_model(model_name,
                       pred_detector,
                       image_seq,
                       pred_window,
                       out_seq=out_seq,
                       device=device,
                       extra_fc=[],
                       hidden_size_rec=hidden_size_rec,
                       num_layers_rec=num_layers_rec,
                       seqlen_rec=seqlen_rec)

    loss_plot_train, mse_plot_valid, mae_plot_valid = train_model(model_name, model, train_set, valid_set, mean_torch, stddev_torch)
    mse_plot_test, mae_plot_test = test_model(model_name, model, 'best_observer.pt', test_set)
    print_results(result_folder, target, loss_plot_train, mse_plot_valid, mae_plot_valid, mse_plot_test, mae_plot_test)