import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from congestion_predict.data_sets import STImageDataset2
from congestion_predict.models import NN
import time

# Running the program
train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
train_set = STImageDataset2(train_data_file_name)
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set)-100000], generator=torch.Generator().manual_seed(5))
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
val_test_set = STImageDataset2(val_test_data_file_name)
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
plt.savefig("resultados/NN/train_mse.png")

flatList_test1 = [item for elem in loss_plot_test for item in elem]
plt.figure(2)
plt.plot(flatList_test1)
plt.title("Testing MSE")
plt.ylabel("MSE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig("resultados/NN/test_mse.png")

flatList_test2 = [item for elem in loss_plot_test2 for item in elem]
plt.figure(3)
plt.plot(flatList_test2)
plt.title("Testing MAE")
plt.ylabel("MAE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig("resultados/NN/test_mae.png")
plt.show()

# sys.stdout.close()