# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from congestion_predict.data_sets import STImgSeqDataset
from congestion_predict.models import eRCNNSeqLin
from congestion_predict.utilities import count_parameters
import congestion_predict.plot as plt_util
import time
import json

# Variables Initialization
train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
label_conf = 'all_lin'
target = 2  # target: 0-Flow, 1-Occ, 2-Speed, 3-All
pred_window = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
# device = "cpu"
epochs = 10
batch_size = 50  # Training Batch size
patience = 5
result_folder = 'resultados/eRCNN/eRCNNSeqLin_fc_layer_test/'


#%% Datasets Creation
cali_dataset_2015 = pd.read_csv(train_data_file_name)
print(cali_dataset_2015.head())
print(cali_dataset_2015.describe())

train_set = STImgSeqDataset(train_data_file_name, label_conf=label_conf, target=target)
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set) - 100000],
                                                 generator=torch.Generator().manual_seed(5))
val_test_set = STImgSeqDataset(val_test_data_file_name, label_conf=label_conf, target=target)
valid_set, test_set, extra = torch.utils.data.random_split(val_test_set, [50000, 50000, len(val_test_set) - 100000],
                                                           generator=torch.Generator().manual_seed(5))

print(f"Size of train_set = {len(train_set)}")
print(f"Size of valid_set = {len(valid_set)}")
print(f"Size of test_set = {len(test_set)}")

#%% View a data sample
image_seq, label = valid_set[0]
print(image_seq.shape)
print(image_seq[0].max())
print(image_seq[0].mean())
# print(image[0])
print(label.shape)
# print(label)

#%% Configure the model
if label_conf == 'all' or label_conf == 'all_lin':
    detectors_pred = 27
else:
    detectors_pred = 1
out_size = detectors_pred * pred_window
hid_error_size = 6 * out_size

best_loss = 1000000
extra_fc_list = [[], [128], [256], [256, 128]]
for extra_fc in extra_fc_list:
    e_rcnn = eRCNNSeqLin(image_seq.shape[1], hid_error_size, out_size, fc_pre_outs=extra_fc, dev=device)
    count_parameters(e_rcnn)
    print(f"Testing the ercnn with {extra_fc} extra FC layers")
    ## Training eRCNN
    # Define Dataloader
    torch.manual_seed(50)  # all exactly the same (model parameters initialization and data split)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    e_rcnn.to(device)  # Put the network on GPU if present

    criterion = nn.MSELoss()  # L2 Norm
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(e_rcnn.parameters(), lr=1e-3)  # ADAM with lr=10^-4
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)  # exponential decay every epoch = 2000iter

    # sys.stdout = open("log.txt", "w")
    torch.cuda.empty_cache()
    min_loss = 100000
    loss_plot_train = []
    mse_plot_valid = []
    mae_plot_valid = []
    for epoch in range(epochs):  # 10 epochs
        print(f"******************Epoch {epoch}*******************\n\n")
        torch.autograd.set_detect_anomaly(True)
        losses_train = []

        # Train
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.permute(1, 0, 2, 3, 4)
            targets = targets.permute(1, 0, 2)
            # print(inputs[0][0][0][0][:10])
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Zero the gradients
            loss = torch.zeros(1, requires_grad=True)

            outputs = e_rcnn(inputs, targets)
            loss = criterion(outputs, targets[-1])
            # print(loss)
            loss.backward()

            nn.utils.clip_grad_norm_(e_rcnn.parameters(), 40)  # gradient clipping norm for eRCNN
            optimizer.step()  # Updated the weights
            losses_train.append(loss.item())
            end = time.time()

            if (batch_idx + 1) % 100 == 0:
                # print(losses_train)
                print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses_train), end - start))
                loss_plot_train.append(losses_train)
                losses_train = []
                start = time.time()
        scheduler.step()

        # Evaluate
        e_rcnn.eval()
        mse_valid = []
        mae_valid = []
        with torch.no_grad():
            for batch_idx, (inputs_valid, targets_valid) in enumerate(valid_loader):
                inputs_valid = inputs_valid.permute(1, 0, 2, 3, 4)
                targets_valid = targets_valid.permute(1, 0, 2)
                inputs_valid, targets_valid = inputs_valid.to(device), targets_valid.to(device)

                outputs_valid = e_rcnn(inputs_valid, targets_valid)
                loss_mse = criterion(outputs_valid, targets_valid[-1])
                loss_mae = criterion2(outputs_valid, targets_valid[-1])
                mse_valid.append(loss_mse.item())
                mae_valid.append(loss_mae.item())
                if (batch_idx + 1) % 100 == 0:
                    print('Batch Index : %d MSE : %.3f' % (batch_idx, np.mean(mse_valid)))
                    print('Batch Index : %d MAE : %.3f' % (batch_idx, np.mean(mae_valid)))
                    mse_plot_valid.append(mse_valid)
                    mae_plot_valid.append(mae_valid)
                    mse_valid = []
                    mae_valid = []
        print('--------------------------------------------------------------')
        if np.mean(mse_plot_valid[-10:]) < min_loss:
            min_loss = np.mean(mse_plot_valid[-10:])
            no_better = 0
            print('Saving best model\n')
            torch.save(e_rcnn.state_dict(), result_folder + f'eRCNN_state_dict_model_{extra_fc}_{target}.pt')
        else:
            no_better += 1
            if no_better >= patience:
                print('Finishing by Early Stopping')
                break
        e_rcnn.train()

    with open(result_folder + f'loss_plot_train_{extra_fc}_{target}.txt', 'w') as filehandle:
        json.dump(loss_plot_train, filehandle)
    with open(result_folder + f'mse_plot_valid_{extra_fc}_{target}.txt', 'w') as filehandle:
        json.dump(mse_plot_valid, filehandle)
    with open(result_folder + f'mae_plot_valid_{extra_fc}_{target}.txt', 'w') as filehandle:
        json.dump(mae_plot_valid, filehandle)
    with open(result_folder + f'final_results_{extra_fc}_{target}.txt', 'w') as filehandle:
        filehandle.write(f"Final Training LOSS = {np.mean(loss_plot_train[-10:])}\n\n")
        filehandle.write(f"Final Validation MSE = {np.mean(mse_plot_valid[-10:])}\n")
        filehandle.write(f"Final Validation MAE = {np.mean(mae_plot_valid[-10:])}\n\n")

    if np.mean(mae_plot_valid) < best_loss:
        with open(result_folder + f'best_model.txt', 'w') as filehandle:
            filehandle.write(f"best model is extra_fc {best_extra_fc}")
        best_loss = np.mean(mae_plot_valid)
        best_extra_fc = extra_fc
        best_model = f"ercnn_{extra_fc}"
    print(f"best model is extra_fc {best_extra_fc}")