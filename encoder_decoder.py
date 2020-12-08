import torch
import torch.nn as nn
import torch.optim as optim
from congestion_predict.data_sets import STEncDecSeqDataset
from congestion_predict.models import EncoderDecoder
import numpy as np
import pandas as pd
from congestion_predict.utilities import count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder = 'resultados/EncoderDecoder/'
folder_results = folder + 'Results/'
torch.manual_seed(50)  # all exactly the same (model parameters initialization and data split)

data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
data = pd.read_csv(data_file_name)
data = data.to_numpy()
mean = np.mean(data, axis=0)[2:]
stddev = np.std(data, axis=0)[2:]

train_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
train_set = STEncDecSeqDataset(train_data_file_name, mean, stddev)
train_set, extra = torch.utils.data.random_split(train_set, [100000, len(train_set)-100000], generator=torch.Generator().manual_seed(5))
val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
val_test_set = STEncDecSeqDataset(val_test_data_file_name, mean, stddev)
valid_set, test_set, extra = torch.utils.data.random_split(val_test_set, [50000, 50000, len(val_test_set)-100000], generator=torch.Generator().manual_seed(5))
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


#%% Model
n_inputs = 27*3
n_outputs = 27
seqlen_conv = 72
seqlen_rec = 12
hidden_size_rec = 50
num_layers_rec = 2

encod_decod = EncoderDecoder(n_inputs=n_inputs, n_outputs=n_outputs, seqlen_conv=seqlen_conv, hidden_size=hidden_size_rec,
                            num_layers=num_layers_rec).to(device)

encod_decod = encod_decod.float()
count_parameters(encod_decod)
#%% Training
batch_size = 50
patience = 5
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss()  # L2 Norm
criterion2 = nn.L1Loss()
optimizer = optim.Adam(encod_decod.parameters(), lr=1e-4)  # ADAM with lr=10^-4
min_loss = 100000

loss_list_train = []
loss_list_test = []
loss_list_test2 = []
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
        loss_acc2 = 0
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
            loss2 = criterion2(Y_pred, Y)
            its += 1
            loss_acc += loss.item()
            loss_acc2 += loss2.item()

    loss_list_test.append(loss_acc / its)
    loss_list_test2.append(loss_acc2 / its)

    print('\nEpoch: {} | Eval MSE: {}\n'.format(epoch, loss_acc / its))
    print('\nEpoch: {} | Eval MAE: {}\n'.format(epoch, loss_acc2 / its))

    # saving loss
    torch.save([loss_list_train, loss_list_test, loss_list_test2], folder_results + 'loss.pt')
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