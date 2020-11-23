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
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        self.mean = np.mean(self.data, axis=0)[2:]
        self.stddev = np.std(self.data, axis=0)[2:]

        if data_size == 0:
            data_size = len(np.unique(self.data[:, 0])) - image_size - pred_window
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:(data_size + image_size + pred_window) * self.detect_num]

        self.var_num = self.data.shape[1] - 2
        self.image_size = image_size
        self.data_size = data_size
        self.pred_window = pred_window
        self.transforms = transforms

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = self.data[idx * self.detect_num:(idx + self.image_size) * self.detect_num, 2:]
        image = torch.from_numpy(image.astype(np.float32))
        image = (image - self.mean) / self.stddev
        image = torch.reshape(image, (self.image_size, self.detect_num, -1))
        image = image.permute(2, 1, 0)
        label = self.data[((idx + self.image_size + self.pred_window) * self.detect_num) + int(self.detect_num / 2), 2:]
        label = torch.from_numpy(label.astype(np.float32))

        if self.transforms:
            image = self.transforms(image)

        return image, label


# %% md

## Creating the CNN Model
class NN(nn.Module):
    def __init__(self, in_sensors=3, height=27, width=72):
        super(NN, self).__init__()

        # Create blocks
        self.linear1 = nn.Linear(in_sensors * height * width, 2048)
        self.drop = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear_out = nn.Linear(1024, 1)

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = x.view(x.size(0), -1)
        #print(out.shape)
        out = nn.ReLU()(self.linear1(out))
        out = self.drop(out)
        out = nn.ReLU()(self.linear2(out))
        out = self.linear_out(out)
        return out

#%%
# Running the program
train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
train_set = STImageDataset(train_data_file_name)
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set)-100000], generator=torch.Generator().manual_seed(5))
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
val_test_set = STImageDataset(val_test_data_file_name)
valid_set, test_set, extra = torch.utils.data.random_split(val_test_set, [50000, 50000, len(val_test_set)-100000], generator=torch.Generator().manual_seed(5))
print(f"Size of train_set = {len(train_set)}")
print(f"Size of valid_set = {len(valid_set)}")
print(f"Size of test_set = {len(test_set)}")

# %%

image, label = train_set[0]
print(image.shape)
print(image[0])
print(image[0].max())
print(image[0].mean())
print(label)
print(label.shape)

#%%
for i in range(3):
    image, label = train_set[i]
    img = (image[2] / image[2].max())
    plt.imshow(img)
    plt.show()


nnModel = NN(3, 27, 72)
nnModel = nnModel.float()

#%%
## Training the CNN
# Define Dataloader
batch_size = 50
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.

nnModel.to(device)  # Put the network on GPU if present

criterion = nn.MSELoss()  # L2 Norm
criterion2 = nn.L1Loss()
optimizer = optim.Adam(nnModel.parameters(), lr=1e-3)  # ADAM with lr=10^-4
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

        outputs = nnModel(inputs.float())  # Forward pass
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
    nnModel.eval()
    total = 0
    losses_test = []
    losses_test2 = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            targets = targets[:, 2]
            targets = torch.unsqueeze(targets, 1)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = nnModel(inputs.float())  # Forward pass
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
    nnModel.train()

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