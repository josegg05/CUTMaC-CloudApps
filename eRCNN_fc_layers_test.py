# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time
import json
import sys


class STImgSeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_name, label_conf='mid', target=3, seq_size=72, image_size=72, data_size=0,
                 pred_window=3,
                 transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        self.mean = np.mean(self.data, axis=0)[2:]
        self.stddev = np.std(self.data, axis=0)[2:]

        if data_size == 0:
            data_size = len(np.unique(self.data[:, 0])) - seq_size - image_size - pred_window
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:(data_size + seq_size + image_size + pred_window) * self.detect_num]

        self.var_num = self.data.shape[1] - 2
        self.seq_size = seq_size
        self.image_size = image_size
        self.data_size = data_size
        self.pred_window = pred_window
        self.transforms = transforms
        self.label_conf = label_conf
        self.target = target

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_seq = []
        labels_seq = []
        for sq in range(self.seq_size):
            image = self.data[(idx + sq) * self.detect_num:(idx + sq + self.image_size) * self.detect_num, 2:]
            image = torch.from_numpy(image.astype(np.float32))
            image = torch.reshape(image, (self.image_size, self.detect_num, -1))
            image = (image - self.mean) / self.stddev
            image = image.permute(2, 1, 0)
            # image = (image-image.mean())/image.std()
            image.unsqueeze_(0)
            if self.target != 3:
                if self.label_conf == 'mid':
                    label = self.data[
                        ((idx + sq + self.image_size + self.pred_window) * self.detect_num) + int(self.detect_num / 2),
                        2 + self.target]
                    label = np.array(label)
                    label = torch.from_numpy(label.astype(np.float32))
                    label.unsqueeze_(-1)
                elif self.label_conf == 'all':
                    label = self.data[
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num) + self.detect_num,
                            2 + self.target]
                    label = torch.from_numpy(label.astype(np.float32))
                else:
                    label = self.data[
                        ((idx + sq + self.image_size + self.pred_window) * self.detect_num) + int(self.label_conf),
                        2 + self.target]
                    label = np.array(label)
                    label = torch.from_numpy(label.astype(np.float32))
                    label.unsqueeze_(-1)
            else:
                if self.label_conf == 'mid':
                    label = self.data[
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num) + int(
                                self.detect_num / 2),
                            2:]
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.label_conf == 'all':
                    label = self.data[
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num) + self.detect_num,
                            2:]
                    label = torch.from_numpy(label.astype(np.float32))
                    label = torch.reshape(label, (1, -1))
                    label.squeeze_()
                else:
                    label = self.data[
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num) + int(self.label_conf),
                            2:]
                    label = torch.from_numpy(label.astype(np.float32))
            # print(f'The label shape is:{label.shape}')
            label.unsqueeze_(0)
            image_seq.append(image)
            labels_seq.append(label)
        image_seq = torch.cat(image_seq,
                              out=torch.Tensor(self.seq_size, self.var_num, self.detect_num, self.image_size))
        labels_seq = torch.cat(labels_seq)
        if self.transforms:
            image_seq = self.transforms(image_seq)

        return image_seq, labels_seq


class eRCNN(nn.Module):
    def __init__(self, input_size, hid_error_size, output_size, n_fc=0, fc_outs=[256]):
        super().__init__()

        self.hid_error_size = hid_error_size
        self.n_fc = n_fc
        last_in = 256 + 32
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1
        )
        self.lin_input = nn.Linear(12 * 35 * 32, 256)  # 32 (25*70) Feature maps after AvgPool2d(2)
        self.lin_error = nn.Linear(hid_error_size, 32)

        if self.n_fc >= 1:
            self.lin1 = nn.Linear(last_in, fc_outs[0])
            last_in = fc_outs[0]
        if self.n_fc >= 2:
            self.lin2 = nn.Linear(last_in, fc_outs[1])
            last_in = fc_outs[1]
        if self.n_fc >= 3:
            self.lin3 = nn.Linear(last_in, fc_outs[2])
            last_in = fc_outs[2]
        self.lin_out = nn.Linear(last_in, output_size)

    def forward(self, input, error):
        out_in = nn.ReLU()(self.conv(input))
        out_in = nn.AvgPool2d(2)(out_in)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
        out_in = out_in.view(-1, self.num_flat_features(out_in))
        out_in = nn.ReLU()(self.lin_input(out_in))
        out_err = nn.ReLU()(self.lin_error(error))
        output = torch.cat((out_in, out_err), 1)
        if self.n_fc >= 1:
            output = nn.ReLU()(self.lin1(output))
        if self.n_fc >= 2:
            output = nn.ReLU()(self.lin2(output))
        if self.n_fc >= 3:
            output = nn.ReLU()(self.lin3(output))
        output = self.lin_out(output)

        return output

    def initError(self, batch_size):
        return torch.zeros(batch_size, self.hid_error_size)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


cali_dataset_2015 = pd.read_csv("datasets/california_paper_eRCNN/I5-N-3/2015.csv")
print(cali_dataset_2015.head())
print(cali_dataset_2015.describe())

label_conf = 'all'
target = 2  # target: 0-Flow, 1-Occ, 2-Speed, 3-All
n_fc = 0  # number of extra fully connected layers
fc_outputs_mult = 1  # multiplier for the number of output of the FC layers
fc_outputs = [128 * fc_outputs_mult, 64 * fc_outputs_mult, 32 * fc_outputs_mult]  # number of outputs of the FC Layers

train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
train_set = STImgSeqDataset(train_data_file_name, label_conf=label_conf,
                            target=target)  # target: 0-Flow, 1-Occ, 2-Speed, 3-All
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set) - 100000],
                                                 generator=torch.Generator().manual_seed(5))
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
val_test_set = STImgSeqDataset(val_test_data_file_name, label_conf=label_conf, target=target)
valid_set, test_set, extra = torch.utils.data.random_split(val_test_set, [50000, 50000, len(val_test_set) - 100000],
                                                           generator=torch.Generator().manual_seed(5))

print(f"Size of train_set = {len(train_set)}")
print(f"Size of valid_set = {len(valid_set)}")
print(f"Size of test_set = {len(test_set)}")

# %%
image, label = valid_set[0]
# print(image.shape)
# print(image[0])
# print(image[0].max())
# print(image[0].mean())
print(label.shape)
print(label)

# %%

if label_conf == 'all':
    detectors_pred = 27
else:
    detectors_pred = 1
hid_error_size = 6 * detectors_pred
out = 1 * detectors_pred

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
# device = "cpu"

cont = 2
r = 0
while r < 3:
    r += 1
    if r == 4:
        r = 1
        cont -= 1
    if cont == 0:
        break
    n_fc = r
    fc_outputs_mult = cont
    fc_outputs = [128 * fc_outputs_mult, 64 * fc_outputs_mult, 32 * fc_outputs_mult]

    e_rcnn = eRCNN(3, hid_error_size, out, n_fc, fc_outputs)
    print(f"Testing the ercnn with {n_fc} extra FC layers and {fc_outputs_mult} multiplier")
    ## Training eRCNN
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

    # sys.stdout = open("log.txt", "w")
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
            # print(targets.shape)
            # targets = targets[:, :, 2]  # no va ahora
            # targets = torch.unsqueeze(targets, 2) # no va ahora
            # print(targets.shape)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Zero the gradients

            error = e_rcnn.initError(batch_size)
            error = error.to(device)
            # print(inputs.shape)
            # print(inputs[i].shape)
            # print(targets.shape)
            # print(error.shape)
            # e_rcnn.zero_grad()
            loss = torch.zeros(1, requires_grad=True)
            for i in range(inputs.shape[0]):
                outputs = e_rcnn(inputs[i], error.detach())
                err_i = outputs - targets[i]
                error = torch.cat((error[:, detectors_pred:], err_i), 1)
                # loss = loss + criterion(outputs, targets[i])  # Loss function Option 2

            # loss = criterion(outputs, targets[-1])    # Compute the Loss
            # loss /= inputs.shape[0]
            loss = criterion(outputs, targets[i])
            loss.backward()
            # print(f"BTT of Batch: {batch_idx}")# Compute the Gradients

            nn.utils.clip_grad_norm_(e_rcnn.parameters(), 40)  # gradient cliping norm for ercnn
            optimizer.step()  # Updated the weights
            losses.append(loss.item())
            end = time.time()

            if batch_idx % 100 == 0:
                print(losses)
                print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses), end - start))
                loss_plot_train.append(np.mean(losses))
                losses = []
                start = time.time()
        scheduler.step()
        torch.save(e_rcnn.state_dict(),
                   f"resultados/fc_layers_test/eRCNN_state_dict_model_{n_fc}_{fc_outputs_mult}_{target}.pt")

        # Evaluate
        e_rcnn.eval()
        total = 0
        losses_test = []
        losses_test2 = []
        with torch.no_grad():
            for batch_idx, (inputs_test, targets_test) in enumerate(valid_loader):
                inputs_test = inputs_test.permute(1, 0, 2, 3, 4)
                targets_test = targets_test.permute(1, 0, 2)
                # targets_test = targets_test[:, :, 2]
                # targets_test = torch.unsqueeze(targets_test, 2)
                inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

                error_test = e_rcnn.initError(batch_size)
                error_test = error_test.to(device)
                loss = torch.zeros(1, requires_grad=True)
                for i in range(inputs_test.shape[0]):
                    outputs_test = e_rcnn(inputs_test[i], error_test.detach())
                    err_i = outputs_test - targets_test[i]
                    error_test = torch.cat((error_test[:, detectors_pred:], err_i), 1)

                loss = criterion(outputs_test, targets_test[i])
                loss2 = criterion2(outputs_test, targets_test[i])
                losses_test.append(loss.item())
                losses_test2.append(loss2.item())
                if batch_idx % 100 == 0:
                    print('Batch Index : %d MSE : %.3f' % (batch_idx, np.mean(losses_test)))
                    print('Batch Index : %d MAE : %.3f' % (batch_idx, np.mean(losses_test2)))
                    loss_plot_test.append(np.mean(losses_test))
                    loss_plot_test2.append(np.mean(losses_test2))
                    # loss_plot_test.append(losses_test)  # uncomment to plot
                    # loss_plot_test2.append(losses_test2)
                    losses_test = []
                    losses_test2 = []
        print('--------------------------------------------------------------')
        e_rcnn.train()

    with open(f'resultados/fc_layers_test/loss_plot_train_{n_fc}_{fc_outputs_mult}_{target}.txt', 'w') as filehandle:
        json.dump(loss_plot_train, filehandle)

    with open(f'resultados/fc_layers_test/loss_plot_test_{n_fc}_{fc_outputs_mult}_{target}.txt', 'w') as filehandle:
        json.dump(loss_plot_test, filehandle)

    with open(f'resultados/fc_layers_test/loss_plot_test2_{n_fc}_{fc_outputs_mult}_{target}.txt', 'w') as filehandle:
        json.dump(loss_plot_test2, filehandle)
