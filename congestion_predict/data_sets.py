import pandas as pd
import numpy as np
import torch

'''
* Image Dataset
'''


# Mean and stddev as input. Label is the "target" value of all detectors --> the best one for image input
class STImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_file_name, mean=None, stddev=None, pred_detector='all', pred_type='solo', pred_window=4, target=2,
                 image_size=72, data_size=0, transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(self.data, axis=0)[2:]
        if stddev is not None:
            self.stddev = stddev
        else:
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
        self.pred_detector = pred_detector
        self.pred_type = pred_type
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

        if self.pred_type == 'solo':
            if self.pred_detector == 'mid':
                label = self.data[
                        ((idx + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(self.detect_num / 2),
                        2 + self.target:(
                                                    2 + self.target) + 1]  # --> this outputs a np.array of shape [1] instead of an int
                # ** Another way without the ":(2 + self.target)+1"
                # label = np.array([label])
                # ** Another way
                # label = np.array(label)
                # label = torch.from_numpy(label.astype(np.float32))
                # label.unsqueeze_(-1)
            elif self.pred_detector == 'all':
                label = self.data[
                        ((idx + self.image_size + (self.pred_window - 1)) * self.detect_num):
                        ((idx + self.image_size + self.pred_window) * self.detect_num),
                        2 + self.target]
            elif self.pred_detector == 'all_iter':  # target.shape --> (seq_size + pred_window, detect_num)
                labels_seq = []
                for wind in range(self.pred_window):
                    label = self.data[
                            ((idx + self.image_size + wind) * self.detect_num):
                            ((idx + self.image_size + wind + 1) * self.detect_num),
                            2 + self.target]
                    label = torch.from_numpy(label.astype(np.float32))
                    label.unsqueeze_(0)
                    labels_seq.append(label)
                    if wind == range(self.pred_window)[-1]:  # last window
                        label = torch.cat(labels_seq)
                        seq = True
        elif self.pred_type == 'mean':
            label_list = []
            if self.pred_detector == 'mid':
                for wind in range(self.pred_window):
                    label = self.data[
                            ((idx + self.image_size + wind) * self.detect_num) + int(self.detect_num / 2),
                            2 + self.target:(2 + self.target) + 1]
                    label_list.append(label)
                label = np.array([np.mean(label_list)])
            elif self.pred_detector == 'all':
                for wind in range(self.pred_window):
                    label = self.data[
                            ((idx + self.image_size + wind) * self.detect_num):
                            ((idx + self.image_size + wind + 1) * self.detect_num),
                            2 + self.target]
                    label_list.append(label)
                label = np.mean(label_list, 0)

        if not seq:
            label = torch.from_numpy(label.astype(np.float32))
            label = torch.unsqueeze(label, 1)

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def get_mean(self):
        m = np.mean(self.data, axis=0)[2:]
        s = np.std(self.data, axis=0)[2:]
        return m, s


# Mean and stddev calculated from the input dataset. Label is all values of the middle detector
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


# Mean and stddev as input. Label is all values of the middle detector
class STImageDataset3(torch.utils.data.Dataset):
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


'''
* Image Sequence Dataset
'''


class STImgSeqDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_file_name, mean=None, stddev=None, pred_detector='all', pred_type='solo', pred_window=4, target=2,
                 seq_size=72, image_size=72, target_norm=False, data_size=0, transforms=None):

        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(self.data, axis=0)[2:]
        if stddev is not None:
            self.stddev = stddev
        else:
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
        self.pred_detector = pred_detector
        self.pred_type = pred_type
        self.target = target
        self.target_norm = target_norm

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
            if self.pred_type == 'solo':
                if self.pred_detector == 'mid':  # target.shape --> (seq_size, 1)
                    label = self.data[
                        ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                            self.detect_num / 2),
                        2 + self.target]
                    label = np.array([label])
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.pred_detector == 'all':  # target.shape --> (seq_size, detect_num)
                    label = self.data[
                            ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                            2 + self.target]
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.pred_detector == 'all_lin':  # target.shape --> (seq_size, detect_num*pred_window)
                    label = self.data[
                            ((idx + sq + self.image_size) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                            2 + self.target]
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.pred_detector == 'all_iter':  # target.shape --> (seq_size + pred_window, detect_num)
                    if sq != range(self.seq_size)[-1]:  # not the last sq
                        label = self.data[
                                ((idx + sq + self.image_size) * self.detect_num):
                                ((idx + sq + self.image_size + 1) * self.detect_num),
                                2 + self.target]
                        label = torch.from_numpy(label.astype(np.float32))
                    else:
                        for wind in range(self.pred_window):
                            label = self.data[
                                    ((idx + sq + self.image_size + wind) * self.detect_num):
                                    ((idx + sq + self.image_size + wind + 1) * self.detect_num),
                                    2 + self.target]
                            label = torch.from_numpy(label.astype(np.float32))
                            if wind != range(self.pred_window)[-1]:  # not last window
                                if (self.target_norm):
                                    label = (label - self.mean[self.target]) / self.stddev[self.target]
                                label.unsqueeze_(0)
                                labels_seq.append(label)
                else:  # target.shape --> (seq_size, 1)
                    label = self.data[
                        ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                            self.pred_detector),
                        2 + self.target]
                    label = np.array(label)
                    label = torch.from_numpy(label.astype(np.float32))
                    label.unsqueeze_(-1)
            elif self.pred_type == 'mean':
                label_list = []
                if self.pred_detector == 'mid':  # target.shape --> (seq_size, 1)
                    for wind in range(self.pred_window):
                        label = self.data[
                                ((idx + sq + self.image_size + wind) * self.detect_num) + int(self.detect_num / 2),
                                2 + self.target:(2 + self.target) + 1]
                        label_list.append(label)
                    label = np.array([np.mean(label_list)])
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.pred_detector == 'all':  # target.shape --> (seq_size, detect_num)
                    for wind in range(self.pred_window):
                        label = self.data[
                                ((idx + sq + self.image_size + wind) * self.detect_num):
                                ((idx + sq + self.image_size + wind + 1) * self.detect_num),
                                2 + self.target]
                        label_list.append(label)
                    label = np.mean(label_list, 0)
                    label = torch.from_numpy(label.astype(np.float32))

            # print(f'The label shape is:{label.shape}')
            if (self.target_norm):
                label = (label - self.mean[self.target]) / self.stddev[self.target]
            label.unsqueeze_(0)
            image_seq.append(image)
            labels_seq.append(label)
        image_seq = torch.cat(image_seq,
                              out=torch.Tensor(self.seq_size, self.var_num, self.detect_num, self.image_size))
        labels_seq = torch.cat(labels_seq)
        if self.transforms:
            image_seq = self.transforms(image_seq)

        return image_seq, labels_seq


class STImgSeqDatasetDayTests(torch.utils.data.Dataset):  # Days divided in 4 periods of 6 hours
    def __init__(self,
                 data_file_name, mean=None, stddev=None, pred_detector='all', pred_type='solo', pred_window=4, target=2,
                 seq_size=72, image_size=72, data_size=0, day_period=0, transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(self.data, axis=0)[2:]
        if stddev is not None:
            self.stddev = stddev
        else:
            self.stddev = np.std(self.data, axis=0)[2:]

        if data_size == 0:
            data_size = int((len(np.unique(self.data[:, 0])) - (image_size - 1) - (seq_size - 1) - pred_window) / 4)
            # print(len(np.unique(self.data[:,0])))
        else:
            self.data = self.data[:(data_size + (image_size - 1) + (seq_size - 1) + pred_window) * self.detect_num]

        self.var_num = self.data.shape[1] - 2
        self.seq_size = seq_size
        self.image_size = image_size
        self.data_size = data_size
        self.pred_window = pred_window
        self.transforms = transforms
        self.pred_detector = pred_detector
        self.pred_type = pred_type
        self.day_period = day_period
        self.target = target

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_seq = []
        labels_seq = []
        offset_multi = idx // 72  # --> (24 h/d * 60 m/h / 5 m/step) = 288 step/d / 4 p/d) = 72 step/p; p = period
        offset_sum = idx % 72
        idx = (offset_multi * 288) + offset_sum + (72 * self.day_period)
        for sq in range(self.seq_size):
            image = self.data[(idx + sq) * self.detect_num:(idx + sq + self.image_size) * self.detect_num, 2:]
            image = torch.from_numpy(image.astype(np.float32))
            image = torch.reshape(image, (self.image_size, self.detect_num, -1))
            image = (image - self.mean) / self.stddev
            image = image.permute(2, 1, 0)
            # image = (image-image.mean())/image.std()
            image.unsqueeze_(0)

            if self.pred_detector == 'all':  # target.shape --> (seq_size, detect_num)
                label = self.data[
                        ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num):
                        ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                        2 + self.target]
                label = torch.from_numpy(label.astype(np.float32))
            elif self.pred_detector == 'all_lin':  # target.shape --> (seq_size, detect_num*pred_window)
                label = self.data[
                        ((idx + sq + self.image_size) * self.detect_num):
                        ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                        2 + self.target]
                label = torch.from_numpy(label.astype(np.float32))
            elif self.pred_detector == 'all_iter':  # target.shape --> (seq_size + pred_window, detect_num)
                if sq != range(self.seq_size)[-1]:  # not the last sq
                    label = self.data[
                            ((idx + sq + self.image_size) * self.detect_num):
                            ((idx + sq + self.image_size + 1) * self.detect_num),
                            2 + self.target]
                    label = torch.from_numpy(label.astype(np.float32))
                else:
                    for wind in range(self.pred_window):
                        label = self.data[
                                ((idx + sq + self.image_size + wind) * self.detect_num):
                                ((idx + sq + self.image_size + wind + 1) * self.detect_num),
                                2 + self.target]
                        label = torch.from_numpy(label.astype(np.float32))
                        if wind != range(self.pred_window)[-1]:  # not last window
                            label.unsqueeze_(0)
                            labels_seq.append(label)
            else:  # target.shape --> (seq_size, 1)
                label = self.data[
                    ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(self.pred_detector),
                    2 + self.target]
                label = np.array(label)
                label = torch.from_numpy(label.astype(np.float32))
                label.unsqueeze_(-1)

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


'''
* Image Sequence Dataset for MTER_LA dataset
'''


class STImgSeqDatasetMTER_LA(torch.utils.data.Dataset):
    def __init__(self,
                 data_file, mean=None, stddev=None, pred_detector='all', pred_type='solo', pred_window=3, target=2,
                 seq_size=12, transforms=None):
        self.data = data_file
        self.detect_num = int(self.data['x'].shape[2])
        if mean is not None:
            self.mean = mean
        else:
            self.mean = self.data['x'][..., 0].mean()
        if stddev is not None:
            self.stddev = stddev
        else:
            self.stddev = self.data['x'][..., 0].std()

        self.var_num = self.data['x'].shape[-1] - 1
        self.seq_size = seq_size
        self.image_size = self.data['x'].shape[1]
        self.data_size = self.data['x'].shape[0] - (seq_size - 1)
        self.pred_window = pred_window
        self.transforms = transforms
        self.pred_detector = pred_detector
        self.pred_type = pred_type
        self.target = target

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_seq = []
        labels_seq = []
        for sq in range(self.seq_size):
            image = self.data['x'][(idx + sq),:,:,0:1]
            image = torch.from_numpy(image.astype(np.float32))
            #image = torch.reshape(image, (self.image_size, self.detect_num, -1))
            image = (image - self.mean) / self.stddev
            image = image.permute(2, 1, 0)
            # image = (image-image.mean())/image.std()
            image.unsqueeze_(0)

            if self.pred_detector == 'all':  # target.shape --> (seq_size, detect_num)
                label = self.data['y'][(idx + sq), self.pred_window - 1,:,0]
                label = torch.from_numpy(label.astype(np.float32))
            elif self.pred_detector == 'all_lin':  # target.shape --> (seq_size, detect_num*pred_window)
                label = self.data['y'][(idx + sq),:self.pred_window,:,0]
                label = torch.from_numpy(label.astype(np.float32))
                label = torch.flatten(label)
            elif self.pred_detector == 'all_iter':  # target.shape --> (seq_size + pred_window, detect_num)
                if sq != range(self.seq_size)[-1]:  # not the last sq
                    label = self.data['y'][(idx + sq), 0,:,0]
                    label = torch.from_numpy(label.astype(np.float32))
                else:
                    for wind in range(self.pred_window):
                        label = self.data['y'][(idx + sq), wind,:,0]
                        label = torch.from_numpy(label.astype(np.float32))
                        if wind != range(self.pred_window)[-1]:  # not last window
                            label.unsqueeze_(0)
                            labels_seq.append(label)

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


class STImgSeqDataset_old(torch.utils.data.Dataset):
    def __init__(self, data_file_name, pred_detector='mid', target=3, seq_size=72, image_size=72, data_size=0,
                 pred_window=4,
                 mean=None,
                 stddev=None,
                 transforms=None):
        self.data = pd.read_csv(data_file_name)
        self.data = self.data.to_numpy()
        self.detect_num = int(np.max(self.data[:, 1]) + 1)
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(self.data, axis=0)[2:]
        if stddev is not None:
            self.stddev = stddev
        else:
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
        self.pred_detector = pred_detector
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
                if self.pred_detector == 'mid':  # target.shape --> (seq_size, 1)
                    label = self.data[
                        ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                            self.detect_num / 2),
                        2 + self.target]
                    label = np.array([label])
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.pred_detector == 'all':  # target.shape --> (seq_size, detect_num)
                    label = self.data[
                            ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                            2 + self.target]
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.pred_detector == 'all_lin':  # target.shape --> (seq_size, detect_num*pred_window)
                    label = self.data[
                            ((idx + sq + self.image_size) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                            2 + self.target]
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.pred_detector == 'all_iter':  # target.shape --> (seq_size + pred_window, detect_num)
                    if sq != range(self.seq_size)[-1]:  # not the last sq
                        label = self.data[
                                ((idx + sq + self.image_size) * self.detect_num):
                                ((idx + sq + self.image_size + 1) * self.detect_num),
                                2 + self.target]
                        label = torch.from_numpy(label.astype(np.float32))
                    else:
                        for wind in range(self.pred_window):
                            label = self.data[
                                    ((idx + sq + self.image_size + wind) * self.detect_num):
                                    ((idx + sq + self.image_size + wind + 1) * self.detect_num),
                                    2 + self.target]
                            label = torch.from_numpy(label.astype(np.float32))
                            if wind != range(self.pred_window)[-1]:  # not last window
                                label.unsqueeze_(0)
                                labels_seq.append(label)

                else:  # target.shape --> (seq_size, 1)
                    label = self.data[
                        ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                            self.pred_detector),
                        2 + self.target]
                    label = np.array(label)
                    label = torch.from_numpy(label.astype(np.float32))
                    label.unsqueeze_(-1)
            else:
                if self.pred_detector == 'mid':
                    label = self.data[
                            ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                                self.detect_num / 2),
                            2:]
                    label = torch.from_numpy(label.astype(np.float32))
                elif self.pred_detector == 'all':
                    label = self.data[
                            ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                            2:]
                    label = torch.from_numpy(label.astype(np.float32))
                    label = torch.reshape(label, (1, -1))
                    label.squeeze_()
                elif self.pred_detector == 'all_lin':  # target.shape --> (seq_size, detect_num*pred_window)
                    label = self.data[
                            ((idx + sq + self.image_size) * self.detect_num):
                            ((idx + sq + self.image_size + self.pred_window) * self.detect_num),
                            2:]
                    label = torch.from_numpy(label.astype(np.float32))
                    label = torch.reshape(label, (1, -1))
                    label.squeeze_()
                else:
                    label = self.data[
                            ((idx + sq + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(
                                self.pred_detector),
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


# Secuence for the EncDec 1D
class STEncDecSeqDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_file_name, mean=None, stddev=None, pred_detector='all', pred_type='solo', pred_window=4, target=2,
                 image_size=72, data_size=0, transforms=None):
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
        self.pred_detector = pred_detector
        self.pred_type = pred_type
        self.target = target

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = self.data[idx * self.detect_num:(idx + self.image_size) * self.detect_num, 2:]
        image = torch.from_numpy(image.astype(np.float32))
        image = (image - self.mean) / self.stddev
        image = image.reshape(self.image_size, -1)

        if self.pred_type == 'solo':
            if self.pred_detector == 'mid':
                label = self.data[
                        ((idx + self.image_size + (self.pred_window - 1)) * self.detect_num) + int(self.detect_num / 2),
                        2 + self.target:(
                                                    2 + self.target) + 1]  # --> this outputs a np.array of shape [1] instead of an int
            elif self.pred_detector == 'all':
                label = self.data[
                        ((idx + self.image_size + (self.pred_window - 1)) * self.detect_num):
                        ((idx + self.image_size + self.pred_window) * self.detect_num),
                        2 + self.target]
        elif self.pred_type == 'mean':
            label_list = []
            if self.pred_detector == 'mid':
                for wind in range(self.pred_window):
                    label = self.data[
                            ((idx + self.image_size + wind) * self.detect_num) + int(self.detect_num / 2),
                            2 + self.target:(2 + self.target) + 1]
                    label_list.append(label)
                label = np.array([np.mean(label_list)])
            elif self.pred_detector == 'all':
                for wind in range(self.pred_window):
                    label = self.data[
                            ((idx + self.image_size + wind) * self.detect_num):
                            ((idx + self.image_size + wind + 1) * self.detect_num),
                            2 + self.target]
                    label_list.append(label)
                label = np.mean(label_list, 0)
        label = torch.from_numpy(label.astype(np.float32))

        if self.transforms:
            image = self.transforms(image)

        return image, label
