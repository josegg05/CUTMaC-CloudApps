import pandas as pd
import numpy as np
from congestion_predict.data_sets import STImgSeqDataset
import torch


dataset_idx = 3
pred_variable = 3
model_idx = 3

# Dataset Variables
datasets_list = ['cali_i5', 'metr_la', 'vegas_i15']
dataset = datasets_list[int(dataset_idx) - 1]

# Data Variables
target = int(pred_variable) - 1
variables_list = ['flow', 'occupancy', 'speed']
pred_window = 4  # 12 for 5min resolution
out_seq = pred_window  # Size of the out sequence
pred_type = 'solo'
seq_size = 24  # Best 12 for eREncDec
image_size = 24  # 72 for the cali_i5 dataset
target_norm = False
batch_div = 40  # 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
# device = "cpu"
torch.manual_seed(50)  # all exactly the same (model parameters initialization and data split)

# Model Variables
models_list = ['eRCNNSeqIter', 'eRCNNSeqLin', 'eREncDec']
model_name = models_list[int(model_idx) - 1]
pred_detector_list = ['all_iter', 'all_lin', 'all_iter']
pred_detector = pred_detector_list[int(model_idx) - 1]
result_folder_list = ['resultados/eRCNN/eRCNNSeqIter/',
                      'resultados/eRCNN/eRCNNSeqLin/',
                      'resultados/EncoderDecoder/Results/']
result_folder = result_folder_list[int(model_idx) - 1]
# eREncDec Model Unique Variables
hidden_size_rec = 40  # 7/20/40/50/70 --> best 50
num_layers_rec = 2  # 2/3/4
seqlen_rec = 6  # 8/12

# Training Variables
if model_name == 'eRCNNSeqIter':
    epochs = 10
else:
    epochs = 10000
batch_size = 50  # Training Batch size
patience = 5

print(f'Starting: Dataset: {dataset}, Model: {model_name}, Target: {variables_list[target]}')


data_file_name = "datasets/las_vegas/i15_bugatti/data_evenly_complete.csv"
data = pd.read_csv(data_file_name)

train_data = data.iloc[:int(data.shape[0] / 2), :]

train_data = train_data.to_numpy()

mean = np.mean(train_data[:, 2:].astype(np.float32), axis=0)
stddev = np.std(train_data[:, 2:].astype(np.float32), axis=0)

detect_num = 28

val_test_data_file_name = "C:/Users/jose0/PycharmProjects/Vegas_I15_PP/testing/data/test_15min_dataset_test.csv"
data_new = pd.read_csv(val_test_data_file_name)
# val_test_set_new = STImgSeqDataset(val_test_data_file_name, mean=mean, stddev=stddev, pred_detector=pred_detector,
#                                        pred_type=pred_type, pred_window=pred_window, target=target,
#                                        seq_size=seq_size, image_size=image_size, target_norm=target_norm,
#                                        detect_num=detect_num)

val_test_data_file_name = "datasets/las_vegas/i15_bugatti/data_evenly_complete_val_test.csv"
data_old = pd.read_csv(val_test_data_file_name)
# val_test_set_old = STImgSeqDataset(val_test_data_file_name, mean=mean, stddev=stddev, pred_detector=pred_detector,
#                                        pred_type=pred_type, pred_window=pred_window, target=target,
#                                        seq_size=seq_size, image_size=image_size, target_norm=target_norm,
#                                        detect_num=detect_num)

print(data_new.head())
print(data_old.head())

print(data_new[['Flow', 'Occupancy', 'Speed']].mean())
print(data_old[['Flow', 'Occupancy', 'Speed']].mean())
print(data_new[['Flow', 'Occupancy', 'Speed']].std())
print(data_old[['Flow', 'Occupancy', 'Speed']].std())
