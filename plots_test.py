import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import json

class STImgSeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_name, label_conf='mid', target=3, seq_size=72, image_size=72, data_size=0, pred_window=3,
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
                            2+self.target]
                    label = np.array(label)
                    label = torch.from_numpy(label.astype(np.float32))
                    label.unsqueeze_(-1)
                elif self.label_conf == 'all':
                    label = self.data[
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num) + self.detect_num,
                            2+self.target]
                    label = torch.from_numpy(label.astype(np.float32))
                else:
                    label = self.data[
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num) + int(self.label_conf),
                            2+self.target]
                    label = np.array(label)
                    label = torch.from_numpy(label.astype(np.float32))
                    label.unsqueeze_(-1)
            else:
                if self.label_conf == 'mid':
                    label = self.data[
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num) + int(self.detect_num / 2),
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
    def __init__(self, input_size, hid_error_size, output_size):
        super().__init__()

        self.hid_error_size = hid_error_size

        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1
        )
        self.lin_input = nn.Linear(12 * 35 * 32, 256)  # 32 (25*70) Feature maps after AvgPool2d(2)
        self.lin_error = nn.Linear(hid_error_size, 32)
        self.lin_out = nn.Linear(256 + 32, output_size)

    def forward(self, input, error):
        out_in = nn.ReLU()(self.conv(input))
        out_in = nn.AvgPool2d(2)(out_in)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
        out_in = out_in.view(-1, self.num_flat_features(out_in))
        out_in = nn.ReLU()(self.lin_input(out_in))
        out_err = nn.ReLU()(self.lin_error(error))
        combined = torch.cat((out_in, out_err), 1)
        output = self.lin_out(combined)

        return output

    def initError(self, batch_size):
        return torch.zeros(batch_size, self.hid_error_size)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class eRCNNSeq(nn.Module):
    def __init__(self, input_size, hid_error_size, output_size, out_seq=1, dev="cpu"):
        super().__init__()

        self.hid_error_size = hid_error_size
        self.out_seq = out_seq
        self.dev = dev
        last_in = 256+32
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1
        )
        self.lin_input = nn.Linear(12 * 35 * 32, 256)  # 32 (25*70) Feature maps after AvgPool2d(2)
        self.lin_error = nn.Linear(hid_error_size, 32)
        self.lin_out = nn.Linear(last_in, output_size)

    def forward(self, input, target):
        error = self.initError(input.shape[1])
        error = error.to(self.dev)
        out_list = []
        for seq in range(input.shape[0]):
            out_in = nn.ReLU()(self.conv(input[seq]))
            out_in = nn.AvgPool2d(2)(out_in)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
            out_in = out_in.view(-1, self.num_flat_features(out_in))
            out_in = nn.ReLU()(self.lin_input(out_in))
            out_err = nn.ReLU()(self.lin_error(error))
            output = torch.cat((out_in, out_err), 1)
            output = self.lin_out(output)

            err_seq = output - target[seq]
            error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)

            if seq >= input.shape[0] - self.out_seq:
                out_list.append(output)

        output = torch.cat(out_list, 1)
        output = output.view(output.shape[0], -1, self.out_seq)
        return output

    def initError(self, batch_size):
        return torch.zeros(batch_size, self.hid_error_size)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#%%
loss_mse_filename = 'resultados/all_speed/loss_plot_test.txt'
loss_mae_filename = 'resultados/all_speed/loss_plot_test2.txt'
target = 2  # 0=Flow, 1=Occ, 2=Speed

#%%
loss_plot_test = []
loss_plot_test2 = []
# with open(loss_mse_filename, 'r') as filehandle:
#     loss_plot_test = json.load(filehandle)
# with open(loss_mae_filename, 'r') as filehandle:
#     loss_plot_test2 = json.load(filehandle)

import ast
with open(loss_mse_filename, 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = ast.literal_eval(line[:-1])

        # add item to the list
        loss_plot_test.append(currentPlace)

with open(loss_mae_filename, 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = ast.literal_eval(line[:-1])

        # add item to the list
        loss_plot_test2.append(currentPlace)

# Plot the training and testing loss

flatList_test1 = [item for elem in loss_plot_test for item in elem]
plt.figure(1)
plt.plot(flatList_test1)
plt.ylim(0, 40)
plt.title("Testing MSE")
plt.ylabel("MSE")
plt.xlabel("Bacth")
plt.grid()
plt.savefig(f"test_mse_{target}.png")

flatList_test2 = [item for elem in loss_plot_test2 for item in elem]
plt.figure(2)
plt.plot(flatList_test2)
plt.ylim(0, 3)
plt.title("Testing MAE")
plt.ylabel("MAE")
plt.xlabel("Bacth")
plt.grid()
plt.savefig(f"test_mae_{target}.png")

flatList_mean_test1 = [np.mean(elem) for elem in loss_plot_test]
plt.figure(3)
plt.plot(flatList_mean_test1)
plt.ylim(0, 10)
plt.title("Testing MSE")
plt.ylabel("MSE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig(f"test_mean_mse_{target}.png")

flatList_mean_test2 = [np.mean(elem) for elem in loss_plot_test2]
plt.figure(4)
plt.plot(flatList_mean_test2)
plt.ylim(0, 2)
plt.title("Testing MAE")
plt.ylabel("MAE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig(f"test_mean_mae_{target}.png")

plt.show()



#%%
# Image Testing
target = 2
label_conf = 'all'
if label_conf == 'all':
    detectors_pred = 27
else:
    detectors_pred = 1
n_hidden = 6 * detectors_pred
out = 1 * detectors_pred
e_rcnn = eRCNNSeq(3, n_hidden, out, out_seq=3)
e_rcnn.load_state_dict(torch.load(f'resultados/eRCNN/sequence/seq3_no_detach/eRCNN_state_dict_model_{target}.pt', map_location=torch.device('cpu')))

seq_size = 72*3
seq_size = 72
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
img_test_set = STImgSeqDataset(val_test_data_file_name, label_conf=label_conf, target=target, seq_size=seq_size)
img_test_set, extra = torch.utils.data.random_split(img_test_set, [100000, len(img_test_set) - 100000],
                                                 generator=torch.Generator().manual_seed(5))
batch_size = 1
valid_len = 200
val_test_set = STImgSeqDataset(val_test_data_file_name, label_conf=label_conf, target=target)
valid_set, test_set, extra = torch.utils.data.random_split(val_test_set, [valid_len, 50000, len(val_test_set) - (valid_len+50000)],
                                                           generator=torch.Generator().manual_seed(5))
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
# device = "cpu"

e_rcnn.to(device)  # Put the network on GPU if present
criterion = nn.MSELoss()  # L2 Norm
criterion2 = nn.L1Loss()

e_rcnn.eval()
#%%
#Good
# losses_test_oneshot = []
# losses_test_oneshot2 = []
# outputs_test_oneshot = []
# targets_test_oneshot = []
# targets_test_oneshot_p = []
# mae_each_sen = [[] for i in range(27)]
# with torch.no_grad():
#     np.random.seed(seed=25)
#     inputs_test, targets_test = img_test_set[np.random.randint(100000)]
#     inputs_test = inputs_test.unsqueeze_(1)
#     targets_test = targets_test.unsqueeze_(1)
#     inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
#     print(inputs_test.shape)
#     print(targets_test.shape)
#     error_test = e_rcnn.initError(1)
#     error_test = error_test.to(device)
#     loss = torch.zeros(1, requires_grad=True)
#     for i in range(inputs_test.shape[0]):
#         outputs_test = e_rcnn(inputs_test[i], error_test.detach())
#         print(f"Out shape = ", outputs_test.shape)
#         print(f"Target shape = ", targets_test[i].shape)
#         for j in range(27):
#             mae_each_sen[j].append(criterion2(outputs_test[0][j], targets_test[i][0][j]))
#         loss = criterion2(outputs_test, targets_test[i-1])
#         loss2 = criterion2(outputs_test, targets_test[i])
#         losses_test_oneshot.append(loss.item())
#         losses_test_oneshot2.append(loss2.item())
#         err_i = outputs_test - targets_test[i]
#         error_test = torch.cat((error_test[:, detectors_pred:], err_i), 1)
#         if i > 40:
#             outputs_test_oneshot.append(outputs_test)
#             targets_test_oneshot.append(targets_test[i])
#             targets_test_oneshot_p.append(targets_test[i - 1])


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
#     for idx in range(inputs_test.shape[0]-72):
#         error_test = e_rcnn.initError(1)
#         error_test = error_test.to(device)
#         loss = torch.zeros(1, requires_grad=True)
#         for i in range(72):
#             outputs_test = e_rcnn(inputs_test[idx + i], error_test.detach())
#             err_i = outputs_test - targets_test[idx + i]
#             error_test = torch.cat((error_test[:, detectors_pred:], err_i), 1)
#         loss = criterion(outputs_test, targets_test[idx + i])
#         loss2 = criterion2(outputs_test, targets_test[idx + i])
#         losses_test_oneshot.append(loss.item())
#         losses_test_oneshot2.append(loss2.item())
#
#         outputs_test_oneshot.append(outputs_test)
#         targets_test_oneshot.append(targets_test[idx + i])


#Good Seq
with torch.no_grad():
    #np.random.seed(seed=25)
    for batch_idx, (inputs_test, targets_test) in enumerate(valid_loader):
        inputs_test = inputs_test.permute(1, 0, 2, 3, 4)
        targets_test = targets_test.permute(1, 0, 2)
        inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

        outputs_test = e_rcnn(inputs_test, targets_test)

        loss = criterion(outputs_test, targets_test[-3:].permute(1, 2, 0))
        loss2 = criterion2(outputs_test, targets_test[-3:].permute(1, 2, 0))

        break

print(f"MSE = {loss}")
print(f"MAE = {loss2}")

plt.figure(1)
plt.title("Prediction image")
plt.ylabel("section")
plt.xlabel("timesteps(5min)")
plt.imshow(outputs_test[0].numpy())
plt.savefig(f"img_seq_out_{target}.png")
plt.show()

plt.figure(2)
plt.title("Traget image")
plt.ylabel("section")
plt.xlabel("timesteps(5min)")
plt.imshow(targets_test[-3:].permute(1, 2, 0)[0].numpy())
plt.savefig(f"img_seq_target_{target}.png")
plt.show()


#Good
losses_test_oneshot = []
losses_test_oneshot2 = []
outputs_test_oneshot = []
targets_test_oneshot = []
targets_test_oneshot_p = []
mae_each_sen = [[] for i in range(27)]
with torch.no_grad():
    #np.random.seed(seed=25)
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

        for j in range(27):
            mae_each_sen[j].append(criterion2(outputs_test[0][j], targets_test[i][0][j]))
        loss = criterion2(outputs_test, targets_test[i-1])
        loss2 = criterion2(outputs_test, targets_test[i])
        losses_test_oneshot.append(loss.item())
        losses_test_oneshot2.append(loss2.item())

        outputs_test_oneshot.append(outputs_test)
        targets_test_oneshot.append(targets_test[i])
        targets_test_oneshot_p.append(targets_test[i-1])

print(f"MAE pre = {np.mean(losses_test_oneshot)}")
print(f"MAE = {np.mean(losses_test_oneshot2)}")
# losses_test_oneshot = []
# losses_test_oneshot2 = []
# outputs_test_oneshot = []
# targets_test_oneshot = []
# targets_test_oneshot_p = []
# with torch.no_grad():
#     #np.random.seed(seed=25)
#     for batch_idx, (inputs_test, targets_test) in enumerate(valid_loader):
#         inputs_test = inputs_test.permute(1, 0, 2, 3, 4)
#         targets_test = targets_test.permute(1, 0, 2)
#         # targets_test = targets_test[:, :, 2]
#         # targets_test = torch.unsqueeze(targets_test, 2)
#         inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
#
#         error_test = e_rcnn.initError(batch_size)
#         error_test = error_test.to(device)
#         loss = torch.zeros(1, requires_grad=True)
#         for i in range(inputs_test.shape[0]):
#             outputs_test = e_rcnn(inputs_test[i], error_test.detach())
#             err_i = outputs_test - targets_test[i]
#             error_test = torch.cat((error_test[:, detectors_pred:], err_i), 1)
#
#             loss = criterion(outputs_test, targets_test[i])
#             loss2 = criterion2(outputs_test, targets_test[i])
#             losses_test_oneshot.append(loss.item())
#             losses_test_oneshot2.append(loss2.item())
#
#             if i > 40:
#                 outputs_test_oneshot.append(outputs_test)
#                 targets_test_oneshot.append(targets_test[i])
#                 targets_test_oneshot_p.append(targets_test[i - 1])
#         break

outputs_test_oneshot = torch.cat(outputs_test_oneshot)
targets_test_oneshot = torch.cat(targets_test_oneshot)
targets_test_oneshot_p = torch.cat(targets_test_oneshot_p)
outputs_test_oneshot = outputs_test_oneshot.permute(1, 0)
targets_test_oneshot = targets_test_oneshot.permute(1, 0)
targets_test_oneshot_p = targets_test_oneshot_p.permute(1, 0)
#print(outputs_test_oneshot)
#print(targets_test_oneshot)
#img_out = (outputs_test_oneshot/ outputs_test_oneshot.max())
#img_target = (targets_test_oneshot/ outputs_test_oneshot.max())
plt.figure(5)
plt.title("Prediction image")
plt.ylabel("section")
plt.xlabel("timesteps(5min)")
plt.imshow(outputs_test_oneshot.cpu())
plt.savefig(f"img_out_{target}.png")
plt.show()
plt.figure(6)
plt.title("Target image")
plt.ylabel("section")
plt.xlabel("timesteps(5min)")
plt.imshow(targets_test_oneshot.cpu())
plt.savefig(f"img_target_{target}.png")
plt.show()
plt.figure(7)
plt.title("Target image pre")
plt.ylabel("section")
plt.xlabel("timesteps(5min)")
plt.imshow(targets_test_oneshot_p.cpu())
plt.savefig(f"img_target_pre_{target}.png")
plt.show()

# print the diff
diff1 = abs(outputs_test_oneshot - targets_test_oneshot)
diff1 = diff1/diff1.max()
diff2 = abs(outputs_test_oneshot - targets_test_oneshot_p)
diff2 = diff2/diff2.max()

plt.figure(10)
plt.title("Differences between images - Target")
plt.ylabel("section")
plt.xlabel("timesteps(5min)")
plt.imshow(torch.ones(diff1.shape) - diff1.cpu(), cmap="gray")
plt.savefig(f"diff_{target}.png")
plt.show()

plt.figure(11)
plt.title("Differences between images - Target[t-1]")
plt.ylabel("section")
plt.xlabel("timesteps(5min)")
plt.imshow(torch.ones(diff2.shape) - diff2.cpu(), cmap="gray")
plt.savefig(f"diff_pre_{target}.png")
plt.show()

#%%
mae_mean_each_sen = [0 for i in range(27)]
for i in range(27):
    mae_mean_each_sen[i] = np.mean(mae_each_sen[i])
print(mae_mean_each_sen)
#%%
plt.figure(8)
plt.plot(losses_test_oneshot)
plt.title("Training MSE")
plt.ylabel("MSE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig(f"train_mse_oneshot_{target}.png")

plt.figure(9)
plt.plot(losses_test_oneshot2)
#plt.ylim(0, 3)
plt.title("Testing MAE")
plt.ylabel("MAE")
plt.xlabel("Bacth")
plt.grid()
plt.savefig(f"test_mae_oneshot_{target}.png")
plt.show()
print(f"MSE = {np.mean(losses_test_oneshot)}")
print(f"MAE = {np.mean(losses_test_oneshot2)}")