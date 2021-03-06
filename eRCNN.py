import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from congestion_predict.data_sets import STImgSeqDataset
from congestion_predict.models import eRCNN
from congestion_predict.utilities import count_parameters
import congestion_predict.evaluation as eval_util
import time
import json

# Variables Initialization
train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"

pred_variable = 'speed'
pred_window = 3
pred_detector = 'mid'
pred_type = 'mean'
error_type = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
# device = "cpu"
epochs = 10
batch_size = 50  # Training Batch size
patience = 5
result_folder = 'resultados/eRCNN/eRCNN/'
torch.manual_seed(50)  # all exactly the same (model parameters initialization and data split)

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)
#%% Datasets Creation
data = pd.read_csv(train_data_file_name)
print(data.head())
print(data.describe())
data = data.to_numpy()
mean = np.mean(data, axis=0)[2:]
stddev = np.std(data, axis=0)[2:]

train_set = STImgSeqDataset(train_data_file_name, mean=mean, stddev=stddev, pred_detector=pred_detector,
                           pred_type=pred_type, pred_window=pred_window, target=target)
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set) - 100000],
                                                 generator=torch.Generator().manual_seed(5))
val_test_set = STImgSeqDataset(val_test_data_file_name, mean=mean, stddev=stddev, pred_detector=pred_detector,
                           pred_type=pred_type, pred_window=pred_window, target=target)
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
print(label)

#%% Create the model
if pred_detector == 'all':
    detectors_pred = 27
else:
    detectors_pred = 1
out_size = 1 * detectors_pred
hid_error_size = 6 * out_size

e_rcnn = eRCNN(image_seq.shape[1], hid_error_size, out_size, pred_window=pred_window, dev=device)
count_parameters(e_rcnn)

#%% Training
# Define Dataloader
# torch.manual_seed(50)  # only the same data splits (not same model parameters initialization)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

e_rcnn.to(device)  # Put the network on GPU if present

criterion = nn.MSELoss()  # L2 Norm
criterion2 = nn.L1Loss()
optimizer = optim.Adam(e_rcnn.parameters(), lr=1e-3)  # ADAM with lr=10^-4
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)  # exponential decay every epoch = 2000iter

#sys.stdout = open("log.txt", "w")
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
        #print(inputs[0][0][0][0][:10])
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the gradients
        loss = torch.zeros(1, requires_grad=True)

        outputs = e_rcnn(inputs, targets)
        #print(outputs.shape)
        loss = criterion(outputs, targets[-1])
        #print(loss)
        loss.backward()

        nn.utils.clip_grad_norm_(e_rcnn.parameters(), 40)  # gradient clipping norm for eRCNN
        optimizer.step()  # Updated the weights
        losses_train.append(loss.item())
        end = time.time()

        if (batch_idx + 1) % 100 == 0:
            #print(losses_train)
            print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses_train), end - start))
            loss_plot_train.append(losses_train)
            losses_train = []
            start = time.time()
    scheduler.step()
    torch.save(e_rcnn.state_dict(), f"resultados/eRCNN/eRCNN_state_dict_model_{target}.pt")

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
        torch.save(e_rcnn.state_dict(), result_folder + 'best_observer.pt')
    else:
        no_better += 1
        if no_better >= patience:
            print('Finishing by Early Stopping')
            break
    e_rcnn.train()

with open(result_folder + f'loss_plot_train_{target}.txt', 'w') as filehandle:
    json.dump(loss_plot_train, filehandle)
with open(result_folder + f'mse_plot_valid_{target}.txt', 'w') as filehandle:
    json.dump(mse_plot_valid, filehandle)
with open(result_folder + f'mae_plot_valid_{target}.txt', 'w') as filehandle:
    json.dump(mae_plot_valid, filehandle)

#%% Testing
print(f"****************** Testing *******************\n\n")
e_rcnn.eval()
mse_plot_test = []
mae_plot_test = []
mse_test = []
mae_test = []
with torch.no_grad():
    for batch_idx, (inputs_test, targets_test) in enumerate(test_loader):
        inputs_test = inputs_test.permute(1, 0, 2, 3, 4)
        targets_test = targets_test.permute(1, 0, 2)
        inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

        outputs_test = e_rcnn(inputs_test, targets_test)
        loss_mse = criterion(outputs_test, targets_test[-1])
        loss_mae = criterion2(outputs_test, targets_test[-1])
        mse_test.append(loss_mse.item())
        mae_test.append(loss_mae.item())
        if (batch_idx + 1) % 100 == 0:
            print('Batch Index : %d MSE : %.3f' % (batch_idx, np.mean(mse_test)))
            print('Batch Index : %d MAE : %.3f' % (batch_idx, np.mean(mae_test)))
            # loss_plot_test.append(np.mean(losses_test))
            # loss_plot_test2.append(np.mean(losses_test2))
            mse_plot_test.append(mse_test)
            mae_plot_test.append(mae_test)
            mse_test = []
            mae_test = []

with open(result_folder + f'mse_plot_test_{target}.txt', 'w') as filehandle:
    json.dump(mse_plot_test, filehandle)
with open(result_folder + f'mae_plot_test_{target}.txt', 'w') as filehandle:
    json.dump(mae_plot_test, filehandle)

#%% Final Results
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

#%% Plotting
eval_util.plot_loss('MSE', f'loss_plot_train_{target}.txt', folder=result_folder)
eval_util.plot_mse(f'mse_plot_valid_{target}.txt', target, folder=result_folder)
eval_util.plot_mae(f'mae_plot_valid_{target}.txt', target, folder=result_folder)
eval_util.plot_mse(f'mse_plot_test_{target}.txt', target, folder=result_folder)
eval_util.plot_mae(f'mae_plot_test_{target}.txt', target, folder=result_folder)
