import torch
import torch.nn as nn
import torch.optim as optim
from congestion_predict.data_sets import STImgSeqDataset
from congestion_predict.data_sets import STImgSeqDatasetMTER_LA
from congestion_predict.models import ErrorEncoderDecoder2D
import numpy as np
import pandas as pd
import time
from congestion_predict.utilities import count_parameters
import congestion_predict.evaluation as eval_util
import json

# Variables Initialization
dataset = 'vegas_i15'  # cali_i5, metr_la, vegas_i15
pred_variable = 'speed'
pred_window = 4
pred_detector = 'all_iter'
pred_type = 'solo'
seq_size = 24
image_size = 72  # for the cali_i5 dataset
target_norm = False
batch_div = 40  # 100

out_seq = pred_window  # Size of the out sequence
seqlen_rec = 6  # 8/12
hidden_size_rec = 40  # 7/20/40/50/70 --> best 50
num_layers_rec = 2  # 2/3/4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10000
batch_size = 50
patience = 5
folder = 'resultados/EncoderDecoder/'
result_folder = folder + 'Results/'
torch.manual_seed(50)  # all exactly the same (model parameters initialization and data split)

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

# %%  Datasets Creation
if dataset == 'cali_i5':
    train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
    val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"

    data = pd.read_csv(train_data_file_name)
    print(data.head())
    print(data.describe())
    data = data.to_numpy()
    mean = np.mean(data, axis=0)[2:]
    stddev = np.std(data, axis=0)[2:]

    train_set = STImgSeqDataset(train_data_file_name, mean=mean, stddev=stddev, pred_detector=pred_detector,
                                pred_type=pred_type, pred_window=pred_window, target=target,
                                seq_size=seq_size, image_size=image_size, target_norm=target_norm)
    train_set, extra1 = torch.utils.data.random_split(train_set, [100000, len(train_set) - 100000],
                                                      generator=torch.Generator().manual_seed(5))
    val_test_set = STImgSeqDataset(val_test_data_file_name, mean=mean, stddev=stddev, pred_detector=pred_detector,
                                   pred_type=pred_type, pred_window=pred_window, target=target,
                                   seq_size=seq_size, image_size=image_size, target_norm=target_norm)
    valid_set, test_set, extra2 = torch.utils.data.random_split(val_test_set,
                                                                [50000, 50000, len(val_test_set) - 100000],
                                                                generator=torch.Generator().manual_seed(5))
    stddev_torch = torch.Tensor([stddev[target]]).to(device)
    mean_torch = torch.Tensor([mean[target]]).to(device)

elif dataset == 'metr_la':
    train_data_file_name = 'datasets/METR-LA/train_filtered_we.npz'
    valid_data_file_name = 'datasets/METR-LA/val_filtered_we.npz'
    test_data_file_name = 'datasets/METR-LA/test_filtered_we.npz'
    train_data_temp = np.load(train_data_file_name)
    train_data = {'x': train_data_temp['x'], 'y': train_data_temp['y']}
    mean = train_data['x'][..., 0].mean()
    stddev = train_data['x'][..., 0].std()
    train_data_temp.close()
    valid_data_temp = np.load(valid_data_file_name)
    valid_data = {'x': valid_data_temp['x'], 'y': valid_data_temp['y']}
    valid_data_temp.close()
    test_data_temp = np.load(test_data_file_name)
    test_data = {'x': test_data_temp['x'], 'y': test_data_temp['y']}
    test_data_temp.close()

    # print(train_data['x'][0,:,:,0:1].shape)
    train_set = STImgSeqDatasetMTER_LA(train_data, pred_detector=pred_detector, seq_size=seq_size,
                                       pred_type=pred_type, pred_window=pred_window, target=target)
    valid_set = STImgSeqDatasetMTER_LA(valid_data, pred_detector=pred_detector, seq_size=seq_size,
                                       pred_type=pred_type, pred_window=pred_window, target=target)
    test_set = STImgSeqDatasetMTER_LA(test_data, pred_detector=pred_detector, seq_size=seq_size,
                                      pred_type=pred_type, pred_window=pred_window, target=target)
    stddev_torch = torch.Tensor([stddev]).to(device)
    mean_torch = torch.Tensor([mean]).to(device)
elif dataset == 'vegas_i15':
    data_file_name = "datasets/las_vegas/i15_bugatti/data_evenly_complete.csv"
    data = pd.read_csv(data_file_name)

    train_data = data.iloc[:int(data.shape[0]/2), :]
    # val_test_data = data.iloc[int(data.shape[0]/2):, :]

    # train_data.to_csv('datasets/las_vegas/i15_bugatti/data_evenly_complete_train.csv', index=False)
    # val_test_data.to_csv('datasets/las_vegas/i15_bugatti/data_evenly_complete_val_test.csv', index=False)

    # print(train_data.head())
    # print(train_data.describe())
    train_data = train_data.to_numpy()

    mean = np.mean(train_data[:, 2:].astype(np.float32), axis=0)
    stddev = np.std(train_data[:, 2:].astype(np.float32), axis=0)
    print(mean)
    print(stddev)

    train_data_file_name = "datasets/las_vegas/i15_bugatti/data_evenly_complete_train.csv"
    val_test_data_file_name = "datasets/las_vegas/i15_bugatti/data_evenly_complete_val_test.csv"
    detect_num = 28

    train_set = STImgSeqDataset(train_data_file_name, mean=mean, stddev=stddev, pred_detector=pred_detector,
                                pred_type=pred_type, pred_window=pred_window, target=target,
                                seq_size=seq_size, image_size=image_size, target_norm=target_norm, detect_num=detect_num)
    val_test_set = STImgSeqDataset(val_test_data_file_name, mean=mean, stddev=stddev, pred_detector=pred_detector,
                                   pred_type=pred_type, pred_window=pred_window, target=target,
                                   seq_size=seq_size, image_size=image_size, target_norm=target_norm, detect_num=detect_num)
    valid_set, test_set = torch.utils.data.random_split(val_test_set,
                                                        [int(len(val_test_set)/2), int(len(val_test_set)/2)],
                                                        generator=torch.Generator().manual_seed(5))
    stddev_torch = torch.Tensor([stddev[target]]).to(device)
    mean_torch = torch.Tensor([mean[target]]).to(device)

print(f"Size of train_set = {len(train_set)}")
print(f"Size of valid_set = {len(valid_set)}")
print(f"Size of test_set = {len(test_set)}")

# %% View Image sample
image_seq, label = train_set[0]
print(image_seq.shape)
print(image_seq[0])
print(image_seq[0].max())
print(image_seq[0].mean())
print(label)
print(label.shape)

# %% Create the model
if 'all' in pred_detector:
    detect_num = image_seq.shape[-2]
else:
    detect_num = 1
image_size = image_seq.shape[-1]
out_size = 1 * detect_num

encod_decod = ErrorEncoderDecoder2D(n_inputs_enc=image_seq.shape[1], n_inputs_dec=detect_num, n_outputs=out_size,
                                    hidden_size=hidden_size_rec, num_layers=num_layers_rec, out_seq=out_seq,
                                    error_size=seqlen_rec, image_size=image_size, dev=device)

encod_decod = encod_decod.float()
count_parameters(encod_decod)
# %% Training
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

encod_decod.to(device)

criterion = nn.MSELoss()  # L2 Norm
criterion2 = nn.L1Loss()
optimizer = optim.Adam(encod_decod.parameters(), lr=1e-4)  # ADAM with lr=10^-4

min_loss = 1000000
loss_plot_train = []
mse_plot_valid = []
mae_plot_valid = []

for epoch in range(epochs):
    print(f"******************Epoch {epoch}*******************\n\n")
    encod_decod.train()
    losses_train = []

    # Train
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.permute(1, 0, 2, 3, 4)
        targets = targets.permute(1, 0, 2)

        inputs, targets = inputs.to(device), targets.to(device)

        # print(inputs.shape)
        # print(targets.shape)

        optimizer.zero_grad()
        loss = torch.zeros(1, requires_grad=True)

        Y_pred = encod_decod(inputs.float(), targets.float())
        # print(Y_pred.shape, Y.shape)
        # print(targets[-out_seq:].permute(1, 0, 2)*stddev_torch + mean_torch)
        if (target_norm):
            loss = criterion(Y_pred * stddev_torch + mean_torch,
                             targets[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
        else:
            loss = criterion(Y_pred, targets[-out_seq:].permute(1, 0, 2))
        loss.backward()
        optimizer.step()
        losses_train.append(loss.item())
        end = time.time()

        if (batch_idx + 1) % batch_div == 0:
            # print(losses_train)
            print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses_train), end - start))
            loss_plot_train.append(losses_train)
            losses_train = []
            start = time.time()

    # evaluation step
    encod_decod.eval()
    mse_valid = []
    mae_valid = []
    with torch.no_grad():
        for batch_idx, (inputs_valid, targets_valid) in enumerate(valid_loader):
            inputs_valid = inputs_valid.permute(1, 0, 2, 3, 4)
            targets_valid = targets_valid.permute(1, 0, 2)
            inputs_valid, targets_valid = inputs_valid.to(device), targets_valid.to(device)

            Y_pred_valid = encod_decod(inputs_valid.float(), targets_valid.float())

            if (target_norm):
                loss_mse = criterion(Y_pred_valid * stddev_torch + mean_torch,
                                     targets_valid[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
                loss_mae = criterion2(Y_pred_valid * stddev_torch + mean_torch,
                                      targets_valid[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
            else:
                loss_mse = criterion(Y_pred_valid, targets_valid[-out_seq:].permute(1, 0, 2))
                loss_mae = criterion2(Y_pred_valid, targets_valid[-out_seq:].permute(1, 0, 2))

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
        torch.save(encod_decod.state_dict(), result_folder + 'best_observer.pt')
    else:
        no_better += 1
        if no_better >= patience:
            print('Finishing by Early Stopping')
            break

# saving loss
with open(result_folder + f'loss_plot_train_{target}.txt', 'w') as filehandle:
    json.dump(loss_plot_train, filehandle)
with open(result_folder + f'mse_plot_valid_{target}.txt', 'w') as filehandle:
    json.dump(mse_plot_valid, filehandle)
with open(result_folder + f'mae_plot_valid_{target}.txt', 'w') as filehandle:
    json.dump(mae_plot_valid, filehandle)
print(f"Final Training LOSS in Epoch {epoch} = {np.mean(loss_plot_train[-10:])}")
print(f"Final Validation MSE in Epoch {epoch} = {np.mean(mse_plot_valid[-10:])}")
print(f"Final Validation MAE in Epoch {epoch} = {np.mean(mae_plot_valid[-10:])}")

# %% Testing
print(f"****************** Testing *******************\n\n")
encod_decod.eval()
mse_plot_test = []
mae_plot_test = []
mse_test = []
mae_test = []
with torch.no_grad():
    for batch_idx, (inputs_test, targets_test) in enumerate(test_loader):
        inputs_test = inputs_test.permute(1, 0, 2, 3, 4)
        targets_test = targets_test.permute(1, 0, 2)
        inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

        Y_pred_test = encod_decod(inputs_test.float(), targets_test.float())

        if (target_norm):
            loss_mse = criterion(Y_pred_test * stddev_torch + mean_torch,
                                 targets_test[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
            loss_mae = criterion2(Y_pred_test * stddev_torch + mean_torch,
                                  targets_test[-out_seq:].permute(1, 0, 2) * stddev_torch + mean_torch)
        else:
            loss_mse = criterion(Y_pred_test, targets_test[-out_seq:].permute(1, 0, 2))
            loss_mae = criterion2(Y_pred_test, targets_test[-out_seq:].permute(1, 0, 2))
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
