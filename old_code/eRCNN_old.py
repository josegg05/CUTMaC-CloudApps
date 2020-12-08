import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from congestion_predict.data_sets import STImgSeqDataset
from congestion_predict.models import eRCNNSeq
from congestion_predict.utilities import count_parameters
import time
import json

# Running the program
cali_dataset_2015 = pd.read_csv("datasets/california_paper_eRCNN/I5-N-3/2015.csv")
print(cali_dataset_2015.head())
print(cali_dataset_2015.describe())

label_conf = 'all'
target = 2  # target: 0-Flow, 1-Occ, 2-Speed, 3-All

train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
train_set = STImgSeqDataset(train_data_file_name, label_conf=label_conf, target=target)
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set) - 100000],
                                                 generator=torch.Generator().manual_seed(5))
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
val_test_set = STImgSeqDataset(val_test_data_file_name, label_conf=label_conf, target=target)
valid_set, test_set, extra = torch.utils.data.random_split(val_test_set, [50000, 50000, len(val_test_set) - 100000],
                                                           generator=torch.Generator().manual_seed(5))

print(f"Size of train_set = {len(train_set)}")
print(f"Size of valid_set = {len(valid_set)}")
print(f"Size of test_set = {len(test_set)}")


# %% View a data sample
image, label = valid_set[0]
#print(image.shape)
#print(image[0])
#print(image[0].max())
#print(image[0].mean())
print(label.shape)
print(label)

# %% Create the model
if label_conf == 'all':
    detectors_pred = 27
else:
    detectors_pred = 1
hid_error_size = 6 * detectors_pred
out = 1 * detectors_pred
out_seq = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
# device = "cpu"

e_rcnn = eRCNNSeq(3, hid_error_size, out, out_seq=out_seq, dev=device)
count_parameters(e_rcnn)

#%% Training
# Define Dataloader
batch_size = 50
torch.manual_seed(50)
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
loss_plot_train = []
loss_plot_test = []
loss_plot_test2 = []
for epoch in range(10):  # 10 epochs
    print(f"******************Epoch {epoch}*******************\n\n")
    torch.autograd.set_detect_anomaly(True)
    losses = []

    # Train
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.permute(1, 0, 2, 3, 4)
        targets = targets.permute(1, 0, 2)

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the gradients
        # e_rcnn.zero_grad()

        error = e_rcnn.initError(batch_size)
        error = error.to(device)

        loss = torch.zeros(1, requires_grad=True)
        #print(outputs.shape)
        for i in range(inputs.shape[0]):  # Uncomment for previous version
            outputs = e_rcnn(inputs[i], error.detach())
            err_i = outputs - targets[i]
            error = torch.cat((error[:, detectors_pred:], err_i), 1)
            # loss = loss + criterion(outputs, targets[i])  # Loss function Option 2

        # loss = criterion(outputs, targets[-1])    # Compute the Loss
        # loss /= inputs.shape[0]
        loss = criterion(outputs, targets[-1])  # Uncomment for previous version
        loss.backward()
        # print(f"BTT of Batch: {batch_idx}")# Compute the Gradients

        nn.utils.clip_grad_norm_(e_rcnn.parameters(), 40)  # gradient clipping norm for eRCNN
        optimizer.step()  # Updated the weights
        losses.append(loss.item())
        end = time.time()

        if (batch_idx + 1) % 100 == 0:
            print(losses)
            print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses), end - start))
            loss_plot_train.append(np.mean(losses))
            losses = []
            start = time.time()
    scheduler.step()
    torch.save(e_rcnn.state_dict(), f"resultados/eRCNN/eRCNN_state_dict_model_{target}.pt")

    # Evaluate
    e_rcnn.eval()
    total = 0
    losses_test = []
    losses_test2 = []
    with torch.no_grad():
        for batch_idx, (inputs_test, targets_test) in enumerate(valid_loader):
            inputs_test = inputs_test.permute(1, 0, 2, 3, 4)
            targets_test = targets_test.permute(1, 0, 2)
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

            error_test = e_rcnn.initError(batch_size)
            error_test = error_test.to(device)

            for i in range(inputs_test.shape[0]):
                outputs_test = e_rcnn(inputs_test[i], error_test.detach())
                err_i = outputs_test - targets_test[i]
                error_test = torch.cat((error_test[:, detectors_pred:], err_i), 1)

            loss = criterion(outputs_test, targets_test[-1])
            loss2 = criterion2(outputs_test, targets_test[-1])
            losses_test.append(loss.item())
            losses_test2.append(loss2.item())
            if (batch_idx + 1) % 100 == 0:
                print('Batch Index : %d MSE : %.3f' % (batch_idx, np.mean(losses_test)))
                print('Batch Index : %d MAE : %.3f' % (batch_idx, np.mean(losses_test2)))
                # loss_plot_test.append(np.mean(losses_test))
                # loss_plot_test2.append(np.mean(losses_test2))
                loss_plot_test.append(losses_test)
                loss_plot_test2.append(losses_test2)
                losses_test = []
                losses_test2 = []
    print('--------------------------------------------------------------')
    e_rcnn.train()


with open(f'resultados/eRCNN/loss_plot_train_{target}.txt', 'w') as filehandle:
    json.dump(loss_plot_train, filehandle)

with open(f'resultados/eRCNN/loss_plot_test_{target}.txt', 'w') as filehandle:
    json.dump(loss_plot_test, filehandle)

with open(f'resultados/eRCNN/loss_plot_test2_{target}.txt', 'w') as filehandle:
    json.dump(loss_plot_test2, filehandle)

with open(f'resultados/eRCNN/final_results_{target}.txt', 'w') as filehandle:
    filehandle.write(f"Final Validation MSE = {np.mean(loss_plot_test[-10:])}\n")
    filehandle.write(f"Final Validation MAE = {np.mean(loss_plot_test2[-10:])}")

print(f"Final Validation MSE = {np.mean(loss_plot_test[-10:])}")
print(f"Final Validation MAE = {np.mean(loss_plot_test2[-10:])}")

# Plot the training and testing loss
plt.figure(1)
plt.plot(loss_plot_train)
plt.title("Training MSE")
plt.ylabel("MSE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig(f"resultados/eRCNN/train_mse_{target}.png")

flatList_test1 = [item for elem in loss_plot_test for item in elem]
plt.figure(20)
plt.plot(flatList_test1)
plt.ylim(0, 40)
plt.title("Testing MSE")
plt.ylabel("MSE")
plt.xlabel("Bacth")
plt.grid()
plt.savefig(f"resultados/eRCNN/test_mse_{target}.png")

flatList_test2 = [item for elem in loss_plot_test2 for item in elem]
plt.figure(3)
plt.plot(flatList_test2)
plt.ylim(0, 3)
plt.title("Testing MAE")
plt.ylabel("MAE")
plt.xlabel("Bacth")
plt.grid()
plt.savefig(f"resultados/eRCNN/test_mae_{target}.png")

flatList_mean_test1 = [np.mean(elem) for elem in loss_plot_test]
plt.figure(4)
plt.plot(flatList_mean_test1)
plt.ylim(0, 10)
plt.title("Testing MSE")
plt.ylabel("MSE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig(f"resultados/eRCNN/test_mean_mse_{target}.png")

flatList_mean_test2 = [np.mean(elem) for elem in loss_plot_test2]
plt.figure(5)
plt.plot(flatList_mean_test2)
plt.ylim(0, 2)
plt.title("Testing MAE")
plt.ylabel("MAE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig(f"resultados/eRCNN/test_mean_mae_{target}.png")

plt.show()


#%%
# # Image Testing
# img_test_set = STImgSeqDataset(val_test_data_file_name, label_conf=label_conf, target=target, seq_size=72*3)
# img_test_set, extra = torch.utils.data.random_split(img_test_set, [100000, len(img_test_set) - 100000],
#                                                  generator=torch.Generator().manual_seed(5))
# e_rcnn.eval()
# total = 0
# losses_test_oneshot = []
# losses_test_oneshot2 = []
# outputs_test_oneshot = []
# targets_test_oneshot = []
# with torch.no_grad():
#     np.random.seed(seed=25)
#     inputs_test, targets_test = img_test_set[np.random.randint(100000)]
#     inputs_test = inputs_test.unsqueeze_(1)
#     targets_test = targets_test.unsqueeze_(1)
#     inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
#
#     error_test = e_rcnn.initError(1)
#     error_test = error_test.to(device)
#     loss = torch.zeros(1, requires_grad=True)
#     for i in range(inputs_test.shape[0]):
#         outputs_test = e_rcnn(inputs_test[i], error_test.detach())
#         loss = criterion(outputs_test, targets_test[i])
#         loss2 = criterion2(outputs_test, targets_test[i])
#         losses_test_oneshot.append(loss.item())
#         losses_test_oneshot2.append(loss2.item())
#         err_i = outputs_test - targets_test[i]
#         error_test = torch.cat((error_test[:, detectors_pred:], err_i), 1)
#         if i > 40:
#             outputs_test_oneshot.append(outputs_test)
#             targets_test_oneshot.append(targets_test[i])
# outputs_test_oneshot = torch.cat(outputs_test_oneshot)
# targets_test_oneshot = torch.cat(targets_test_oneshot)
# outputs_test_oneshot = outputs_test_oneshot.permute(1, 0)
# targets_test_oneshot = targets_test_oneshot.permute(1, 0)
# #print(outputs_test_oneshot)
# #print(targets_test_oneshot)
# #img_out = (outputs_test_oneshot/ outputs_test_oneshot.max())
# #img_target = (targets_test_oneshot/ outputs_test_oneshot.max())
# plt.figure()
# plt.imshow(outputs_test_oneshot.cpu())
# plt.savefig(f"img_out_{target}.png")
# plt.show()
# plt.figure()
# plt.imshow(targets_test_oneshot.cpu())
# plt.savefig(f"img_target_{target}.png")
# plt.show()
#
# plt.figure()
# plt.plot(losses_test_oneshot)
# plt.title("Training MSE")
# plt.ylabel("MSE")
# plt.xlabel("Bacthx100")
# plt.grid()
# plt.savefig(f"train_mse_oneshot_{target}.png")
#
# plt.figure()
# plt.plot(losses_test_oneshot2)
# #plt.ylim(0, 3)
# plt.title("Testing MAE")
# plt.ylabel("MAE")
# plt.xlabel("Bacth")
# plt.grid()
# plt.savefig(f"test_mae_oneshot_{target}.png")
#
# #sys.stdout.close()
