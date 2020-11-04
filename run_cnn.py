import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import time


# %%
## Creating Image Dataset
class STImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_name, image_size=72, data_size=0, pred_window=3, transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()

        if data_size == 0:
            data_size = len(np.unique(self.data[:, 0])) - image_size - pred_window
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:data_size + image_size]

        self.detect_num = int(np.max(self.data[:, 1]) + 1)
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
        image = image.permute(2, 1, 0)
        image = (image - image.mean()) / image.std()
        label = self.data[((idx + self.image_size + self.pred_window) * self.detect_num) + int(self.detect_num / 2), 2:]
        label = torch.from_numpy(label.astype(np.float32))

        if self.transforms:
            image = self.transforms(image)

        return image, label


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
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same
        # as it's input, have a convolutional layer downsample the layer
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

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
train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
train_set = STImageDataset(train_data_file_name)
test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
test_set = STImageDataset(test_data_file_name)
print(f"Size of train_set = {len(train_set)}")
print(f"Size of test_set = {len(test_set)}")

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
train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.

model.to(device)  # Put the network on GPU if present

criterion = nn.MSELoss()  # L2 Norm
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # ADAM with lr=10^-4
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)  # exponential decay every epoch = 2000iter

for epoch in range(10):  # 10 epochs
    losses = []

    # Train
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        targets = targets[:, 2]
        targets = torch.unsqueeze(targets, 1)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the gradients

        outputs = model(inputs.float())  # Forward pass
        loss = criterion(outputs, targets)  # Compute the Loss
        loss.backward()  # Compute the Gradients

        # nn.utils.clip_grad_norm(model.parameters(), 40) # gradient cliping norm for ercnn
        optimizer.step()  # Updated the weights
        losses.append(loss.item())
        end = time.time()

        if batch_idx % 100 == 0:
            print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses), end - start))

            start = time.time()
    scheduler.step()

    # Evaluate
    model.eval()
    total = 0
    losses_test = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            targets = targets[2]
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs.float())  # Forward pass
            loss = criterion(outputs, targets)  # Compute the Loss

            losses_test.append(loss.item())

            if batch_idx % 100 == 0:
                print('Batch Index : %d Loss : %.3f' % (batch_idx, np.mean(losses)))

        print('--------------------------------------------------------------')
    model.train()
