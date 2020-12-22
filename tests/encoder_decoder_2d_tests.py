import torch
import torch.nn as nn
import torch.optim as optim
from congestion_predict.data_sets import STImageDataset
from congestion_predict.models import EncoderDecoder2D
import numpy as np
import pandas as pd
import time
from congestion_predict.utilities import count_parameters
import congestion_predict.evaluation as eval_util
import json


pred_variable = 'speed'
pred_detector = 'all'
pred_type = 'solo'  # 'solo'; 'mean'
pred_window = 4
n_inputs_enc = 3  #nm
n_inputs_dec = 3*27  #nm
seqlen_rec = 12  # 6/12/24/36/72 --> best 12~6
hidden_size_rec = 50  # 7/20/40/50/70 --> best 50
num_layers_rec = 2   # 2/3/4 -- best 2
lin1_rec = False  # True/False --> best True
lin1_conv = False; first_lin_size = 128  # False/True; 128/True; 256 --> best
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batch_size = 50
patience = 5
folder = 'resultados/EncoderDecoder/'
test = 'lin1_conv_active'  # num_layers_rec / seqlen_rec / lin1_rec_active
result_folder = folder + 'Results/' + test + '/'
torch.manual_seed(50)  # all exactly the same (model parameters initialization and data split)

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

data_file_name = "../datasets/california_paper_eRCNN/I5-N-3/2015.csv"
data = pd.read_csv(data_file_name)
data = data.to_numpy()
mean = np.mean(data, axis=0)[2:]
stddev = np.std(data, axis=0)[2:]

train_data_file_name = "../datasets/california_paper_eRCNN/I5-N-3/2015.csv"
train_set = STImageDataset(train_data_file_name, mean=mean, stddev=mean, pred_detector=pred_detector,
                            pred_type=pred_type, pred_window=pred_window, target=target)
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set) - 100000],
                                                 generator=torch.Generator().manual_seed(5))
val_test_data_file_name = "../datasets/california_paper_eRCNN/I5-N-3/2016.csv"
val_test_set = STImageDataset(val_test_data_file_name, mean=mean, stddev=mean, pred_detector=pred_detector,
                            pred_type=pred_type, pred_window=pred_window, target=target)
valid_set, test_set, extra = torch.utils.data.random_split(val_test_set, [50000, 50000, len(val_test_set) - 100000],
                                                           generator=torch.Generator().manual_seed(5))
print(f"Size of train_set = {len(train_set)}")
print(f"Size of valid_set = {len(valid_set)}")
print(f"Size of test_set = {len(test_set)}")

# %% View Image sample
image, label = train_set[0]
print(image.shape)
print(image[0])
print(image[0].max())
print(image[0].mean())
print(label)
print(label.shape)

if pred_detector == 'all':
    n_pred_detector = 27
else:
    n_pred_detector = 1
out_size = 1 * n_pred_detector

#seqlen_rec_list = [6]#,12, 24, 36, 72]  # 1
#num_layers_rec_list = [2, 3, 4]  # 2
#lin1_rec_active = [True]  # 3
lin1_conv = True; first_lin_size_list = [128, 256]  # 4
#for seqlen_rec in seqlen_rec_list:  # 1
#for num_layers_rec in num_layers_rec_list:  # 2
#for lin1_rec in lin1_rec_active:  # 3
for first_lin_size in first_lin_size_list:  # 4
    #plot_ind = seqlen_rec  # 1
    #plot_ind = num_layers_rec  # 2
    #plot_ind = 1  # 3
    plot_ind = first_lin_size  # 3
    # Model
    encod_decod = EncoderDecoder2D(n_inputs_enc=n_inputs_enc, n_inputs_dec=n_inputs_dec, n_outputs=out_size,
                                   hidden_size=hidden_size_rec,
                                   num_layers=num_layers_rec, lin1_conv=lin1_conv, lin1_rec=lin1_rec,
                                   first_lin_size=first_lin_size).to(device)

    encod_decod = encod_decod.float()
    count_parameters(encod_decod)
    # %% Training
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()  # L2 Norm
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(encod_decod.parameters(), lr=1e-4)  # ADAM with lr=10^-4

    min_loss = 100000
    loss_plot_train = []
    mse_plot_valid = []
    mae_plot_valid = []
    for epoch in range(epochs):
        print(f"******************Epoch {epoch}*******************\n\n")
        # training step
        encod_decod.train()
        losses_train = []
        start = time.time()
        for batch_idx, (input, targets) in enumerate(train_loader):
            targets = torch.unsqueeze(targets, 1)
            X_r_pre = input.transpose(1, 3)
            X_r_pre = X_r_pre.reshape(X_r_pre.shape[0], X_r_pre.shape[1], -1)
            X_c, X_r, Y = input.to(device), X_r_pre[:, -seqlen_rec:, :].to(device), targets.to(device)

            optimizer.zero_grad()
            Y_pred = encod_decod(X_c.float(), X_r.float())
            # print(Y_pred.shape, Y.shape)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
            losses_train.append(loss.item())
            end = time.time()

            if(batch_idx + 1) % 100 == 0:
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
            for batch_idx, (input, targets) in enumerate(valid_loader):
                targets = torch.unsqueeze(targets, 1)
                X_r_pre = input.transpose(1, 3)
                X_r_pre = X_r_pre.reshape(X_r_pre.shape[0], X_r_pre.shape[1], -1)
                X_c, X_r, Y = input.to(device), X_r_pre[:, -seqlen_rec:, :].to(device), targets.to(device)

                Y_pred = encod_decod(X_c.float(), X_r.float())

                loss_mse = criterion(Y_pred, Y)
                loss_mae = criterion2(Y_pred, Y)
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
        print(f"Final Training LOSS in Epoch {epoch} = {np.mean(loss_plot_train[-10:])}")
        print(f"Final Validation MSE in Epoch {epoch} = {np.mean(mse_plot_valid[-10:])}")
        print(f"Final Validation MAE in Epoch {epoch} = {np.mean(mae_plot_valid[-10:])}\n")

        if np.mean(mse_plot_valid[-10:]) < min_loss:
            min_loss = np.mean(mse_plot_valid[-10:])
            no_better = 0
            print('Saving best model\n')
            torch.save(encod_decod.state_dict(), result_folder + f'best_observer_{plot_ind}.pt')
        else:
            no_better += 1
            if no_better >= patience:
                print('Finishing by Early Stopping')
                break

    # saving loss
    with open(result_folder + f'loss_plot_train_{target}_{plot_ind}.txt', 'w') as filehandle:
        json.dump(loss_plot_train, filehandle)
    with open(result_folder + f'mse_plot_valid_{target}_{plot_ind}.txt', 'w') as filehandle:
        json.dump(mse_plot_valid, filehandle)
    with open(result_folder + f'mae_plot_valid_{target}_{plot_ind}.txt', 'w') as filehandle:
        json.dump(mae_plot_valid, filehandle)
    print(f"Final Training LOSS in Epoch {epoch} = {np.mean(loss_plot_train[-10:])}")
    print(f"Final Validation MSE in Epoch {epoch} = {np.mean(mse_plot_valid[-10:])}")
    print(f"Final Validation MAE in Epoch {epoch} = {np.mean(mae_plot_valid[-10:])}")

    #%% Testing
    print(f"****************** Testing *******************\n\n")
    encod_decod.eval()
    mse_plot_test = []
    mae_plot_test = []
    mse_test = []
    mae_test = []
    with torch.no_grad():
        for batch_idx, (inputs_test, targets_test) in enumerate(test_loader):
            targets_test = torch.unsqueeze(targets_test, 1)
            X_r_pre = inputs_test.transpose(1, 3)
            X_r_pre = X_r_pre.reshape(X_r_pre.shape[0], X_r_pre.shape[1], -1)
            X_c, X_r, Y = inputs_test.to(device), X_r_pre[:, -seqlen_rec:, :].to(device), targets_test.to(device)

            Y_pred = encod_decod(X_c.float(), X_r.float())

            loss_mse = criterion(Y_pred, Y)
            loss_mae = criterion2(Y_pred, Y)
            mse_test.append(loss_mse.item())
            mae_test.append(loss_mae.item())
            if (batch_idx + 1) % 100 == 0:
                print('Batch Index : %d MSE : %.3f' % (batch_idx, np.mean(mse_test)))
                print('Batch Index : %d MAE : %.3f' % (batch_idx, np.mean(mae_test)))
                mse_plot_test.append(mse_test)
                mae_plot_test.append(mae_test)
                mse_test = []
                mae_test = []

    with open(result_folder + f'mse_plot_test_{target}_{plot_ind}.txt', 'w') as filehandle:
        json.dump(mse_plot_test, filehandle)
    with open(result_folder + f'mae_plot_test_{target}_{plot_ind}.txt', 'w') as filehandle:
        json.dump(mae_plot_test, filehandle)

    #%% Final Results
    with open(result_folder + f'final_results_{target}_{plot_ind}.txt', 'w') as filehandle:
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

#%% Plotting
eval_util.plot_loss('MSE', f'loss_plot_train_{target}_{plot_ind}.txt', folder=result_folder)
eval_util.plot_mse(f'mse_plot_valid_{target}_{plot_ind}.txt', target, folder=result_folder)
eval_util.plot_mae(f'mae_plot_valid_{target}_{plot_ind}.txt', target, folder=result_folder)
eval_util.plot_mse(f'mse_plot_test_{target}_{plot_ind}.txt', target, folder=result_folder)
eval_util.plot_mae(f'mae_plot_test_{target}_{plot_ind}.txt', target, folder=result_folder)