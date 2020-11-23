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
