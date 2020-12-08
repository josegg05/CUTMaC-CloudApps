import pandas as pd
import numpy as np
import torch

'''
* Image Dataset
'''


# Mean and stddev as input
class STImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_name, mean, stddev, image_size=72, data_size=0, pred_window=4, transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        # self.mean = np.mean(self.data, axis=0)[2:]
        # self.stddev = np.std(self.data, axis=0)[2:]
        self.mean = mean
        self.stddev = stddev

        if data_size == 0:
            data_size = len(np.unique(self.data[:, 0])) - (image_size - 1) - pred_window
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:(data_size + (image_size - 1) + pred_window) * self.detect_num]

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
        # image = (image - image.mean()) / image.std()  # This is not correct
        label = self.data[
                ((idx + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(self.detect_num / 2), 2:]
        label = torch.from_numpy(label.astype(np.float32))

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def get_mean(self):
        m = np.mean(self.data, axis=0)[2:]
        s = np.std(self.data, axis=0)[2:]
        return m, s


# Mean and stddev calculated from the input dataset
class STImageDataset2(torch.utils.data.Dataset):
    def __init__(self, data_file_name, image_size=72, data_size=0, pred_window=4, transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        self.mean = np.mean(self.data, axis=0)[2:]
        self.stddev = np.std(self.data, axis=0)[2:]

        if data_size == 0:
            data_size = len(np.unique(self.data[:, 0])) - (image_size - 1) - pred_window
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:(data_size + (image_size - 1) + pred_window) * self.detect_num]

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
        # image = (image - image.mean()) / image.std()  # This is not correct
        label = self.data[
                ((idx + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(self.detect_num / 2), 2:]
        label = torch.from_numpy(label.astype(np.float32))

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def get_mean(self):
        m = np.mean(self.data, axis=0)[2:]
        s = np.std(self.data, axis=0)[2:]
        return m, s


class STImageDataset3(torch.utils.data.Dataset):
    def __init__(self, data_file_name, mean, stddev, image_size=72, data_size=0, pred_window=4, target=2,
                 transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        # self.mean = np.mean(self.data, axis=0)[2:]
        # self.stddev = np.std(self.data, axis=0)[2:]
        self.mean = mean
        self.stddev = stddev

        if data_size == 0:
            data_size = len(np.unique(self.data[:, 0])) - (image_size - 1) - pred_window
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:(data_size + (image_size - 1) + pred_window) * self.detect_num]

        self.image_size = image_size
        self.data_size = data_size
        self.pred_window = pred_window
        self.transforms = transforms
        self.target = target

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = self.data[idx * self.detect_num:(idx + self.image_size) * self.detect_num, 2:]
        image = torch.from_numpy(image.astype(np.float32))
        image = torch.reshape(image, (self.image_size, self.detect_num, -1))
        image = (image - self.mean) / self.stddev
        image = image.permute(2, 1, 0)
        # image = (image - image.mean()) / image.std()  # This is not correct
        label = self.data[
                ((idx + self.image_size + (self.pred_window - 1)) * self.detect_num):
                ((idx + self.image_size + self.pred_window) * self.detect_num),
                2 + self.target]
        label = torch.from_numpy(label.astype(np.float32))

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def get_mean(self):
        m = np.mean(self.data, axis=0)[2:]
        s = np.std(self.data, axis=0)[2:]
        return m, s


'''
* Image Sequence Dataset
'''


class STImgSeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_name, label_conf='mid', target=3, seq_size=72, image_size=72, data_size=0,
                 pred_window=4,
                 transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        self.mean = np.mean(self.data, axis=0)[2:]
        self.stddev = np.std(self.data, axis=0)[2:]

        if data_size == 0:
            data_size = len(np.unique(self.data[:, 0])) - (image_size - 1) - (seq_size - 1) - pred_window
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:(data_size + (image_size - 1) + (seq_size - 1) + pred_window) * self.detect_num]

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
                        ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                            self.detect_num / 2),
                        2 + self.target]
                    label = np.array(label)
                    label = torch.from_numpy(label.astype(np.float32))
                    label.unsqueeze_(-1)
                elif self.label_conf == 'all':
                    label = self.data[
                            ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                            2 + self.target]
                    label = torch.from_numpy(label.astype(np.float32))
                else:
                    label = self.data[
                        ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                            self.label_conf),
                        2 + self.target]
                    label = np.array(label)
                    label = torch.from_numpy(label.astype(np.float32))
                    label.unsqueeze_(-1)
            else:
                if self.label_conf == 'mid':
                    label = self.data[
                            ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                                self.detect_num / 2),
                            2:]
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.label_conf == 'all':
                    label = self.data[
                            ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                            2:]
                    label = torch.from_numpy(label.astype(np.float32))
                    label = torch.reshape(label, (1, -1))
                    label.squeeze_()
                else:
                    label = self.data[
                            ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                                self.label_conf),
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


class STEncDecSeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_name, mean, stddev, image_size=72, data_size=0, pred_window=4, target=2,
                 transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        # self.mean = np.mean(self.data, axis=0)[2:]
        # self.stddev = np.std(self.data, axis=0)[2:]
        self.mean = mean
        self.stddev = stddev

        if data_size == 0:
            data_size = len(np.unique(self.data[:, 0])) - (image_size - 1) - pred_window
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:(data_size + (image_size - 1) + pred_window) * self.detect_num]

        self.image_size = image_size
        self.data_size = data_size
        self.pred_window = pred_window
        self.transforms = transforms
        self.target = target

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = self.data[idx * self.detect_num:(idx + self.image_size) * self.detect_num, 2:]
        image = torch.from_numpy(image.astype(np.float32))
        image = (image - self.mean) / self.stddev
        image = image.reshape(self.image_size, -1)
        label = self.data[
                ((idx + self.image_size + (self.pred_window - 1)) * self.detect_num):
                ((idx + self.image_size + self.pred_window) * self.detect_num),
                2 + self.target]
        label = torch.from_numpy(label.astype(np.float32))

        if self.transforms:
            image = self.transforms(image)

        return image, label
