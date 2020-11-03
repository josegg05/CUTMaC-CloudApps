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


