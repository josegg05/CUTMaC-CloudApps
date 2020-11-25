import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from observer import EncoderDecoder2
import os
from pylab import rcParams


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder = 'EncoderDecoder/'
folder_results = folder + 'Results/'


#%% Data
class STImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_name, mean, stddev, image_size=72, data_size=0, pred_window=3, target=2, transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        # self.mean = np.mean(self.data, axis=0)[2:]
        # self.stddev = np.std(self.data, axis=0)[2:]
        self.mean = mean
        self.stddev = stddev

        if data_size == 0:
            data_size = len(np.unique(self.data[:, 0])) - image_size - pred_window
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:(data_size + image_size + pred_window) * self.detect_num]

        self.image_size = image_size
        self.data_size = data_size
        self.pred_window = pred_window
        self.transforms = transforms
        self.target = target

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = self.data[idx * self.detect_num:(idx + self.image_size) * self.detect_num, 2:]
        image = torch.from_numpy(image.astype(np.float32))
        image = (image - self.mean) / self.stddev
        image = image.reshape(self.image_size, -1)
        # image = torch.reshape(image, (self.image_size, self.detect_num, -1))
        # image = (image - self.mean) / self.stddev
        # image = image.view(self.image_size, -1)
        # image = image.permute(2, 1, 0)
        # image = (image - image.mean()) / image.std()
        label = self.data[
                ((idx + self.image_size + self.pred_window) * self.detect_num):
                ((idx + self.image_size + self.pred_window) * self.detect_num) + self.detect_num,
                2 + self.target]
        # label = self.data[((idx + self.image_size + self.pred_window) * self.detect_num) + int(self.detect_num / 2), 2:]
        label = torch.from_numpy(label.astype(np.float32))

        if self.transforms:
            image = self.transforms(image)

        return image, label

data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
data = pd.read_csv(data_file_name)
data = data.to_numpy()
mean = np.mean(data, axis=0)[2:]
stddev = np.std(data, axis=0)[2:]

train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
train_set = STImageDataset(train_data_file_name, mean, stddev)
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set)-100000], generator=torch.Generator().manual_seed(5))
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
val_test_set = STImageDataset(val_test_data_file_name, mean, stddev)
valid_set, test_set, extra = torch.utils.data.random_split(val_test_set, [50000, 50000, len(val_test_set)-100000], generator=torch.Generator().manual_seed(5))
print(f"Size of train_set = {len(train_set)}")
print(f"Size of valid_set = {len(valid_set)}")
print(f"Size of test_set = {len(test_set)}")

image, label = train_set[0]
print(image.shape)
print(image[0])
print(image[0].max())
print(image[0].mean())
print(label)
print(label.shape)


#%% Model
n_inputs = 27*3
n_outputs = 27
seqlen_conv = 72
seqlen_rec = 10
hidden_size_rec = 7
num_layers_rec = 2

encod_decod = EncoderDecoder2(n_inputs=n_inputs, n_outputs=n_outputs, seqlen_conv=seqlen_conv, hidden_size=hidden_size_rec,
                            num_layers=num_layers_rec).to(device)

encod_decod = encod_decod.float()
#%% Training
batch_size = 50
patience = 50
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss()  # L2 Norm
criterion2 = nn.L1Loss()
optimizer = optim.Adam(encod_decod.parameters(), lr=1e-4)  # ADAM with lr=10^-4

loss_list_train = []
loss_list_test = []
epochs = 10
for epoch in range(epochs):
    last_epoch = epoch
    # training step
    encod_decod.train()
    its = 0
    loss_acc = 0
    for batch_idx, (input, targets) in enumerate(train_loader):
        #targets = targets[:, 2]
        targets = torch.unsqueeze(targets, 1)
        X_c, X_r, Y = input.to(device), input[:, -seqlen_rec:, :].to(device), targets.to(device)

        # X = X_train[batch * batch_size: (batch + 1) * batch_size, :, :]
        # X_c = torch.from_numpy(X).float().to(device)
        # X_r = X[:, -seqlen_rec:, :].to(device)
        # Y = torch.from_numpy(Y_train[batch * batch_size: (batch + 1) * batch_size, :]).float().to(
        #     device)

        optimizer.zero_grad()
        Y_pred = encod_decod(X_c.float(), X_r.float())
        #print(Y_pred.shape, Y.shape)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        loss_acc += loss.item()
        its += 1

        if batch_idx % 100 == 0:
            print('Epoch: {} | Training Batch {} | loss:{}'.format(epoch, batch_idx, loss_acc / its))

    loss_list_train.append(loss_acc / its)
    # evaluation step

    encod_decod.eval()

    with torch.no_grad():
        loss_acc = 0
        its = 0
        for batch_idx, (input, targets) in enumerate(valid_loader):
            #targets = targets[:, 2]
            targets = torch.unsqueeze(targets, 1)
            X_c, X_r, Y = input.to(device), input[:, -seqlen_rec:, :].to(device), targets.to(device)

            # X = torch.from_numpy(X_test[batch * batch_size: (batch + 1) * batch_size, :, :])
            # X_c = X.float().to(device)
            # X_r = X[:, -seqlen_rec:, :].float().to(device)
            # Y = torch.from_numpy(Y_test[batch * batch_size: (batch + 1) * batch_size, :]).float().to(
            #     device)

            Y_pred = encod_decod(X_c.float(), X_r.float())

            loss = criterion(Y_pred, Y)
            its += 1
            loss_acc += loss.item()

    loss_list_test.append(loss_acc / its)

    print('\nEpoch: {} | Eval loss: {}\n'.format(epoch, loss_acc / its))

    # saving loss
    torch.save([loss_list_train, loss_list_test], folder_results + 'loss.pt')
    torch.save(encod_decod.state_dict(), folder_results + 'model_observer_{}.pt'.format(epoch))

    if loss_acc / its < min_loss:
        min_loss = loss_acc / its
        no_better = 0
        print('Saving best model\n')
        torch.save(encod_decod.state_dict(), folder + 'best_observer.pt')
    else:
        no_better += 1
        if no_better >= patience:
            print('Finishing by Early Stopping')
            break