import torch

dataset = torch.zeros((72, 128, 3, 27, 30))
for idx, batch in enumerate(dataset):
    print(batch[0].shape)

dataset.shape[0]

hola = torch.zeros(1, 6)
hola.shape
dataset2 = torch.zeros(72, 128, 3, 27, 30)
dataset2.shape

##%

a = np.array([1, 2, 3])
b = np.array([2, 3, 4])

# %%

import pandas as pd
import torch
import numpy as np


class STImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_name, image_size=72, data_size=0, transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()

        if data_size == 0:
            data_size = len(self.data) - image_size
        else:
            self.data = self.data[:data_size + image_size]

        self.detect_num = int(np.max(self.data[:, 1])+1)
        self.image_size = image_size
        self.data_size = data_size
        self.transforms = transforms

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = self.data[idx * self.detect_num:(idx + self.image_size) * self.detect_num, 2:]
        image = torch.from_numpy(image)
        image = torch.reshape(image, (self.image_size, self.detect_num, -1))
        image = image.permute(2, 1, 0)
        label = torch.from_numpy(self.data[((idx + self.image_size + 3) * self.detect_num)+int(self.detect_num/2), 2:])

        if self.transforms:
            image = self.transforms(image)

        return image, label


data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2015.csv"
cali_dataset_2015 = pd.read_csv("datasets/california_paper_eRCNN/I5-N-3/2015.csv")
my_dataset = STImageDataset(data_file_name)

#%%
import matplotlib.pyplot as plt
image, label = my_dataset[0]
print(image.shape)
print(image[0])
print(image[0].max())
print(image[0].mean())
print(label)

for i in range(3):
    image, label = my_dataset[i]
    img = (image[2]/image[2].max())
    plt.imshow(img)
    plt.show()

#%%

image_2=torch.Tensor([[111,112,113],[121,122,123],[211,212,213],[221,222,223],[311,312,313],[321,322,323],[411,412,413],[421,422,423],])
image_2=image_2.view(4, 2, -1)
print(image_2)
image_2=image_2.permute(2,1,0)
print(image_2)

#%%
import torch
a = []
for i in range(10):
    a.append(torch.rand(1, 100, 100))

b = torch.Tensor(10, 100, 100)
print(b.shape)
torch.cat(a, out=b)
print(b.shape)

#%%
import numpy as np

lala=np.array([1,2,3])
lele=np.array([4,5,6])
lolo=np.array([7,8,9])

lista=[]
lista.append(lala)
lista.append(lele)
lista.append(lele)
print(lista)

#%%
import matplotlib.pyplot as plt

lala = [1000,100,30,10,8,20,5,2,1,0]
lolo = [2000,100,30,10,8,20,5,2,1,0]
plt.figure(1)
plt.plot(lala)
plt.title("Training MSE")
plt.ylabel("MSE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig("train_mse.png")

plt.figure(2)
plt.plot(lolo)
plt.title("Testing MSE")
plt.ylabel("MSE")
plt.xlabel("Bacthx100")
plt.grid()
plt.savefig("test_mse.png")
plt.show()

#%%
# formas de obtener el índice del menor valor
import torch
validation_error = torch.tensor([20, 18, 3, 35, 5])
print('1: min = ', (validation_error == validation_error.min()).nonzero().item()) # 1
print('2:', validation_error.min(0))
print('min = ', validation_error.min(0)[1].item()) # 2

#%%

import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [[0,1],[2,3],[4,5],[6,7],[8,9],[10]]
    def __len__(self):
        return 8
    def __getitem__(self, idx):
        if idx % 2 == 0:
            return [self.data[int(idx / 2)][0], self.data[int(idx / 2)][1]]
        else:
            return [self.data[int(idx / 2)][1], self.data[int((idx / 2)) + 1][0]]

s=Dataset()
print(s[0])
print(s[1])
valid_s, test_s, extra = torch.utils.data.random_split(s, [4, 2, len(s)-6], generator=torch.Generator().manual_seed(42))

print(len(valid_s))



valid_loader= torch.utils.data.DataLoader(valid_s,
                                     batch_size=1, shuffle=False,
                                     num_workers=0)
test_loader= torch.utils.data.DataLoader(test_s,
                                     batch_size=1, shuffle=False,
                                     num_workers=0)
for batch_idx, inputs in enumerate(valid_loader):
    print(batch_idx," and ", inputs)

for batch_idx, inputs in enumerate(test_loader):
    print(batch_idx, " and ", inputs)

#%%
import json

# define list with values
basicList = [[1, 2], "Cape Town", 4.6]

# open output file for writing
with open('listfile.txt', 'w') as filehandle:
    json.dump(basicList, filehandle)

# open output file for reading
with open('listfile.txt', 'r') as filehandle:
    basicList2 = json.load(filehandle)

print((basicList2[0][1]))

#%%
#Padding
import torch
lacaca = torch.Tensor([[[[1,2,3], [4,5,6]]]])
print(lacaca.shape)
pad = (0,0,0,0,14,15,0,0)
#lacacapad = torch.nn.functional.pad(lacaca, pad, mode='constant', value=0)
m = torch.nn.ZeroPad2d((0,0,0,0,14,15,0,0))
lacacapad = m(lacaca)
print(lacacapad.shape)

#%%
a = [0,1,2,3,4,5,6,7,8,9]
print(a[-2:])

import torch
h = torch.rand([64, 30, 5])
print(h.transpose(0, 1).transpose(0, 2).shape)

#%% batch norm
import torch
from torch import nn

class my_model(nn.Module):
    def __init__(self, in_channels):
        super(my_model, self).__init__()
        self.bn_norm = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        out = self.bn_norm(x)
        return out


bn = my_model(3)
for _ in range(10):
    x = torch.randn(1, 3, 24, 24)
    out = bn(x)
print(bn.bn_norm.running_mean)
print(bn.bn_norm.running_var)

bn.eval()
for _ in range(10):
    x = torch.randn(10, 3, 24, 24)
    out = bn(x)
print(bn.bn_norm.running_mean)
print(bn.bn_norm.running_var)

#%%
cont = 0
i = 0
while i < 5:
    print(i)
    # code

    i += 1
    if i == 4:
        i = 0
        cont += 1
    if cont == 3:
        print("salimos")
        break

#%% Test de FC layers
import json
import numpy as np
loss = []
loss2 = []
# open output file for reading
for i in range(3):
    filename = f"resultados/fc_layers_test/loss_plot_test_{i+1}_1_2.txt"
    with open(filename, 'r') as filehandle:
        loss.append(json.load(filehandle))

    print(f"MSE of {i+1} FC layer with multiplier 1 = {np.mean(loss[i][-10:])}")

for i in range(3):
    filename = f"resultados/fc_layers_test/loss_plot_test2_{i+1}_1_2.txt"
    with open(filename, 'r') as filehandle:
        loss2.append(json.load(filehandle))

    print(f"MAE of {i+1} FC layer with multiplier 1 = {np.mean(loss2[i][-10:])}")

loss = []
loss2 = []
for i in range(3):
    filename = f"resultados/fc_layers_test/loss_plot_test_{i+1}_2_2.txt"
    with open(filename, 'r') as filehandle:
        loss.append(json.load(filehandle))

    print(f"MSE of {i+1} FC layer with multiplier 2 = {np.mean(loss[i][-10:])}")

for i in range(3):
    filename = f"resultados/fc_layers_test/loss_plot_test2_{i+1}_2_2.txt"
    with open(filename, 'r') as filehandle:
        loss2.append(json.load(filehandle))

    print(f"MAE of {i+1} FC layer with multiplier 2 = {np.mean(loss2[i][-10:])}")


#%%

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class log_regression(nn.Module):
    def __init__(self, input, lin=False, t = 0):
        super(log_regression, self).__init__()
        self.lin=lin
        self.linear = nn.Linear(input, 1)
        self.treshold = nn.Threshold(t, 1)

    def forward(self, x):
        if self.lin:
            out = (self.linear(x) > 0.5).float()
            print(out)
        else:
            out = torch.sigmoid(self.linear(x))
        return out

log_model = log_regression(1, True)
x = torch.Tensor([i for i in range(-100, 101) if i != 0])
#x = torch.Tensor([6,8,10,12, 15,17,19,21])
y = torch.Tensor([0 if i < 0 else 1 for i in range(-100, 100)])
#y = torch.Tensor([0,0,0,0,1,1,1,1])
X = x.view(200, -1)   # (200, -1)
criterion = nn.L1Loss()
optimizer = optim.SGD(log_model.parameters(), lr=0.01)
for epoch in range(100):
    for sample in range(x.shape[0]):
        y_hat = log_model(x[sample].view(1, -1))
        loss = criterion(y_hat, y[sample].view(1, -1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


with torch.no_grad():
    plt.plot(x.numpy(), log_model(X).numpy())
    plt.plot(x.numpy(), y.numpy(), 'bo')
    plt.show()

#%% Test Resultados Ecod Decod
import torch
folder_list = ['hidSize_7/kernel_9-7-5/', 'hidSize_7/kernel_7-5-3/', 'hidSize_7/conv_3_desactivada/',
               'hidSize_20/kernel_9-7-5/', 'hidSize_20/conv_3_desactivada/',
               'hidSize_40/kernel_9-7-5/', 'hidSize_40/conv_3_desactivada/',
               'hidSize_50/conv_3_desactivada/']

folder = 'resultados/EncoderDecoder/'
for i in range(len(folder_list)):
    folder_results = folder + folder_list[i]
    loss = torch.load(folder_results + 'loss.pt')

    print(folder_list[i])
    print(f"MSE = {loss[1][-1]}")
    try:
        print(f"MAE = {loss[2][-1]}")
    except:
        print("no MAE")

#%%
import torch
import torch.nn as nn
list = [torch.Tensor([[1,1,1,1,1,1,1], [4,4,4,4,4,4,4]]), torch.Tensor([[2,2,2,2,2,2,2], [5,5,5,5,5,5,5]]), torch.Tensor([[3,3,3,3,3,3,3], [6,6,6,6,6,6,6]])]
list = [torch.Tensor([[3,3,3,3,3,3,3], [6,6,6,6,6,6,6]])]
a = torch.cat(list, 1)
b = a.view(a.shape[0], -1, 1)
c = a.view(a.shape[0], 1, -1)

c = c.transpose(1,2)

print(b)
print(c)

target1 = torch.Tensor([[[2], [2], [2], [2], [2], [2], [2]], [[2], [2], [2], [2], [2], [2], [2]]])
target2 = torch.Tensor([[2,2,2,2,2,2,2],[2,2,2,2,2,2,2]])
criterion = nn.MSELoss()

loss1 = criterion(c, target1)
loss2 = criterion(c.squeeze(), target2)
print(loss1)
print(loss2)

#%%
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
test_predictions = torch.Tensor(np.load('resultados/CNN/Tensorflow/70_30/test_prediction_1.npy'))
test_target = torch.Tensor(np.load('resultados/CNN/Tensorflow/70_30/test_labels_1.npy'))

criterion = nn.MSELoss()
criterion2 = nn.L1Loss()

loss1 = criterion(test_predictions, test_target)
loss2 = criterion2(test_predictions, test_target)
print(f"CNN Tensorflow MSE = {loss1}")
print(f"CNN Tensorflow MAE = {loss2}")

#%% plot utilityt
import congestion_predict.evaluation as eval_util
target = 2
out_seq = 3
label_conf = 'all'
model_path = f'resultados/eRCNN/sequence/seq3_no_detach/eRCNN_state_dict_model_{target}.pt'
#eval_util.plot_seq_out(target, 'all', out_seq, model_path=model_path)

eval_util.plot_image(target, 'all', out_seq, model_path=model_path)

#%% numpy view
import numpy as np
from prettytable import PrettyTable
x = PrettyTable()
y = np.array([
        ["Adelaide", 1295, 1158259, 600.5],
        ["Brisbane", 5905, 1857594, 1146.4],
        ["Darwin", 112, 120900, 1714.7],
        ["Hobart", 1357, 205556, 619.5],
        ["Sydney", 2058, 4336374, 1214.8],
        ["Melbourne", 1566, 3806092, 646.9],
        ["Perth", 5386, 1554769, 869.4],
    ])
x.field_names = ["City name", "Area", "Population", "Annual Rainfall"]
x.add_rows(y)
print(x)


import numpy as np
import torch
caca = np.array([1,1,1,1,5,1,1,5,1,1])
coco = torch.from_numpy(caca)
cucu = torch.tensor(caca)
caca[caca > 3] = 2
coco[coco < 2] = 0
print(caca)
print(coco)
print(cucu)


import torch
lolo = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32).unsqueeze_(0)
targ = torch.tensor([[4,7,13],[4,5,6]], dtype=torch.float32).unsqueeze_(0)
loss = torch.nn.MSELoss()
print(lolo.shape)
print(loss(lolo, targ))
print(loss(lolo.view(1,-1), targ.view(1,-1)))


import torch
coco = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)
print(torch.mean(coco, (0, 1)))



with open('dog_breeds.txt', 'w') as reader:
    # Read & print the entire file
    reader.write("me cago en esta mierda")

with open('dog_breeds.txt', 'r') as reader:
    # Read & print the entire file
    print(reader.read())


import torch
caca = torch.tensor([[1,3], [3,5]])
coco = torch.zeros(caca.shape)
print(caca)
print(coco)

import random
print(random.choices([True, False], weights=[0.5, 1], k=1))
