import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import time
import random


# %%
## Creating Image Dataset
class STImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_name, mean, stddev, image_size=72, data_size=0, pred_window=3, transforms=None):
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

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = self.data[idx * self.detect_num:(idx + self.image_size) * self.detect_num, 2:]
        image = torch.from_numpy(image.astype(np.float32))
        image = torch.reshape(image, (self.image_size, self.detect_num, -1))
        image = (image - self.mean) / self.stddev
        image = image.permute(2, 1, 0)
        ##image = (image - image.mean()) / image.std()
        label = self.data[((idx + self.image_size + self.pred_window) * self.detect_num) + int(self.detect_num / 2), 2:]
        label = torch.from_numpy(label.astype(np.float32))

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def get_mean(self):
        m = np.mean(self.data, axis=0)[2:]
        s = np.std(self.data, axis=0)[2:]
        return m, s


# %% md

## Creating the CNN Model

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.001, eps=0.001)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.001, eps=0.001)

        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same
        # as it's input, have a convolutional layer downsample the layer
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            diff = out_channels - in_channels
            if diff % 2 == 0:
                pad = (0, 0, 0, 0, int(diff / 2), int(diff / 2), 0, 0)
            else:
                pad = (0, 0, 0, 0, int(diff / 2), int(diff / 2) + 1, 0, 0)
            self.shortcut = nn.ZeroPad2d(pad)
        # if stride != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(
        #             in_channels=in_channels, out_channels=out_channels,
        #             kernel_size=(1, 1), stride=stride, bias=False
        #         ),
        #         nn.BatchNorm2d(out_channels)
        #     )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class CNN(nn.Module):
    def __init__(self, in_channels=3, height=27, width=72):
        super(CNN, self).__init__()

        # Create blocks
        self.block1 = self._create_block(in_channels, 32, stride=1)
        self.block2 = self._create_block(32, 64, stride=1)
        self.block3 = self._create_block(64, 96, stride=1)
        self.linear1 = nn.Linear(96 * height * width, 2048)
        self.drop = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear_out = nn.Linear(1024, 1)

    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(out.size(0), -1)
        out = nn.ReLU()(self.linear1(out))
        out = self.drop(out)
        out = nn.ReLU()(self.linear2(out))
        out = self.linear_out(out)
        return out

#%%
# Running the program
data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
data = pd.read_csv(data_file_name)
data = data.to_numpy()
mean = np.mean(data, axis=0)[2:]
stddev = np.std(data, axis=0)[2:]

train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
train_set = STImageDataset(train_data_file_name, mean, stddev)
print(f"Mean train = {train_set.get_mean()}")
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set)-100000], generator=torch.Generator().manual_seed(50))
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
val_test_set = STImageDataset(val_test_data_file_name, mean, stddev)
print(f"Mean test = {val_test_set.get_mean()}")
valid_set, test_set, extra = torch.utils.data.random_split(val_test_set, [50000, 50000, len(val_test_set)-100000], generator=torch.Generator().manual_seed(50))
print(f"Size of train_set = {len(train_set)}")
print(f"Size of valid_set = {len(valid_set)}")
print(f"Size of test_set = {len(test_set)}")

#%%
# lolo = np.array([0,0,60])
# lala = np.array([1,1,10])
# image, label = train_set[0]
# print(image.shape)
# print(image[2])
# #print(((image - mean)/stddev)[2])
# %%

image, label = train_set[0]
print(image.shape)
print(image[0])
print(image[0].max())
print(image[0].mean())
print(label)

for i in range(3):
    image, label = train_set[i]
    img = (image[2] / image[2].max())
    plt.imshow(img)
    plt.show()


model = CNN(3, 27, 72)
model = model.float()


# %%
## Training the CNN
# Define Dataloader
batch_size = 50
torch.manual_seed(50)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.

model.to(device)  # Put the network on GPU if present

criterion = nn.MSELoss()  # L2 Norm
criterion2 = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # ADAM with lr=10^-4
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)  # exponential decay every epoch = 2000iter

# sys.stdout = open("log.txt", "w")
torch.cuda.empty_cache()
loss_plot_train = []
loss_plot_test = []
loss_plot_test2 = []
for epoch in range(10):  # 10 epochs
    print(f"******************Epoch {epoch}*******************\n\n")
    #torch.autograd.set_detect_anomaly(True)
    losses = []

    # Train
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        targets = targets[:, 2]
        targets = torch.unsqueeze(targets, 1)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the gradients
        loss = torch.zeros(1, requires_grad=True)

        outputs = model(inputs.float())  # Forward pass
        loss = criterion(outputs, targets)  # Compute the Loss
        loss.backward()  # Compute the Gradients

        # nn.utils.clip_grad_norm(model.parameters(), 40) # gradient cliping norm for ercnn
        optimizer.step()  # Updated the weights
        losses.append(loss.item())
        end = time.time()

        if batch_idx % 100 == 0:
            print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses), end - start))
            loss_plot_train.append(np.mean(losses))
            losses = []
            start = time.time()
    scheduler.step()

    # Evaluate
    for j in random.sample(range(0, 50000), 1000):
        model(valid_set[j][0].unsqueeze_(0).to(device).float())
    model.eval()
    losses_test = []
    losses_test2 = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            targets = targets[:, 2]
            targets = torch.unsqueeze(targets, 1)
            inputs, targets = inputs.to(device), targets.to(device)
            loss = torch.zeros(1, requires_grad=True)
            loss2 = torch.zeros(1, requires_grad=True)

            outputs = model(inputs.float())  # Forward pass
            loss = criterion(outputs, targets)  # Compute the Loss
            loss2 = criterion2(outputs, targets)

            losses_test.append(loss.item())
            losses_test2.append(loss2.item())
            if batch_idx % 100 == 0:
                print('Batch Index : %d Loss : %.3f' % (batch_idx, np.mean(losses_test)))
                print('Batch Index : %d MAE : %.3f' % (batch_idx, np.mean(losses_test2)))
                loss_plot_test.append(losses_test)
                loss_plot_test2.append(losses_test2)
                losses_test = []
                losses_test2 = []
        print('--------------------------------------------------------------')
    model.train()

# Plot the training and testing loss
plt.figure(1)
plt.plot(loss_plot_train)
plt.title("Training MSE")
plt.ylabel("MSE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig("train_mse.png")

flatList_test1 = [item for elem in loss_plot_test for item in elem]
plt.figure(2)
plt.plot(flatList_test1)
plt.title("Testing MSE")
plt.ylabel("MSE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig("test_mse.png")

flatList_test2 = [item for elem in loss_plot_test2 for item in elem]
plt.figure(3)
plt.plot(flatList_test2)
plt.title("Testing MAE")
plt.ylabel("MAE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig("test_mae.png")
plt.show()

# sys.stdout.close()