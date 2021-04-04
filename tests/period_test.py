import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from congestion_predict.data_sets import STImgSeqDatasetDayTests
from congestion_predict.models import eRCNNSeqLin, eRCNNSeqIter, ErrorEncoderDecoder2D
import congestion_predict.evaluation as eval_util


# Variables Initialization
train_data_file_name = "../datasets/california_paper_eRCNN/I5-N-3/2015.csv"
test_data_file_name = "../datasets/california_paper_eRCNN/I5-N-3/2016.csv"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 50
batch_size = 500

#%% Data Description
data = pd.read_csv(train_data_file_name)
print(data.head())
print(data.describe())
data = data.to_numpy()
print(data.shape)
data = np.reshape(data, (-1, 27, 5))
data_period = data[:(data.shape[0] - data.shape[0]%72), :, :]
data_period = np.reshape(data_period, (-1, 72, 27, 5))
data_period = data_period[:(data_period.shape[0] - data_period.shape[0]%4), :, :, :]
data_period = np.reshape(data_period, (-1, 4, 72, 27, 5))
print(data_period.shape)
print(data_period[0, 0, 0])
print(data_period[0, 1, 0])
print(data_period[1, 0, 0])
# data_period = np.transpose(data_period, (1, 0, 3, 2, 4))
# print(data_period[0, 0, 0])
# print(data_period[0, 1, 0])
# print(data_period[1, 0, 0])

data_week = data[:(data.shape[0] - data.shape[0]%288), :, :]
data_week = np.reshape(data_week, (-1, 288, 27, 5))
data_week = data_week[:(data_week.shape[0] - data_week.shape[0]%7), :, :, :]
data_week = np.reshape(data_week, (-1, 7, 288, 27, 5))
print(data_week.shape)
print(data_week[0, 0, 0])
print(data_week[0, 1, 0])
print(data_week[1, 0, 0])

#%%
data_period_mean = np.mean(data_period, axis=0)
print(data_period_mean.shape)
print(data_period_mean[0, 0])

for var in range(1, 4):
    complete_day = np.reshape(data_period_mean, (-1, 27, 5))
    image_day = complete_day[:, :, -var]
    maximum = image_day.max()
    image_day = image_day / maximum
    color_min = image_day.min()
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(np.transpose(image_day, (1, 0)), vmin=color_min, vmax=1)
    x_label_list = ['0', '6', '12', '18']
    ax.set_xticks([0, 72, 144, 216])
    ax.set_xticklabels(x_label_list)
    fig.colorbar(img)
    plt.savefig(f"resultados/dayPer_var{var}.png")
    plt.show()

    for per in range(len(data_period_mean)):
        image = data_period_mean[per, :, :, -var]
        image = image / maximum
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(np.transpose(image, (1, 0)), vmin=color_min, vmax=1)

        x_label_list = ['0', '1', '2', '3', '4', '5']
        ax.set_xticks([0, 12, 24, 36, 48, 60])
        ax.set_xticklabels(x_label_list)
        fig.colorbar(img)
        plt.savefig(f"resultados/dayPer_var{var}_per{per}.png")
        plt.show()

#%%
data_week_mean = np.mean(data_week, axis=0)
print(data_week_mean.shape)
print(data_week_mean[0, 0])

for var in range(1, 4):
    complete_day = np.reshape(data_week_mean, (-1, 27, 5))
    image_day = complete_day[:, :, -var]
    maximum = image_day.max()
    image_day = image_day / maximum
    color_min = image_day.min()
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(np.transpose(image_day, (1, 0)), vmin=color_min, vmax=1)
    x_label_list = ['0', '1', '2', '3', '4', '5', '6']
    ax.set_xticks([0, 288, 288*2, 288*3, 288*4, 288*5, 288*6])
    ax.set_xticklabels(x_label_list)
    fig.colorbar(img)
    plt.savefig(f"resultados/weekDay_var{var}.png")
    plt.show()

    for day in range(len(data_week_mean)):
        image = data_week_mean[day, :, :, -var]
        image = image / maximum
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(np.transpose(image, (1, 0)), vmin=color_min, vmax=1)

        x_label_list = ['0', '6', '12', '18']
        ax.set_xticks([0, 72, 144, 216])
        ax.set_xticklabels(x_label_list)
        fig.colorbar(img)
        plt.savefig(f"resultados/weekDay_var{var}_day{day}.png")
        plt.show()


# # %% eRCNNSeqLin
# pred_variable = 'speed'
# pred_window = 3
# pred_detector = 'all_lin'
# pred_type = 'solo'
#
# out_seq = 3
# img_size = 20
#
# variables_list = ['flow', 'occupancy', 'speed']
# target = variables_list.index(pred_variable)
#
# result_folder = 'resultados/eRCNN/eRCNNSeqLin/ev1/[256]'
# model_name = f'eRCNN_state_dict_model_{target}.pt'
#
# if 'all' in pred_detector:
#     detectors_pred = 27
# else:
#     detectors_pred = 1
# out_size = detectors_pred * pred_window
# hid_error_size = 6 * out_size
# model_type = 'eRCNNSeqLin'
# e_rcnn = eRCNNSeqLin(3, hid_error_size, out_size, pred_window, fc_pre_outs=[256], dev=device)
# e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))
#
# day_periods = [0, 1, 2, 3]
# for day_period in day_periods:
#     test_set = STImgSeqDatasetDayTests(test_data_file_name, pred_detector=pred_detector,
#                                pred_window=pred_window, target=target, day_period=day_period)
#     test_set, _, _ = torch.utils.data.random_split(test_set, [12500, 12500, len(test_set) - 25000],
#                                                                generator=torch.Generator().manual_seed(seed))
#     eval_util.loss_evaluation(test_set,
#                               e_rcnn,
#                               model_type,
#                               batch_size=batch_size,
#                               res_folder=result_folder,
#                               file_sfx=f'_period_{day_period}',
#                               device=device,
#                               seed=seed)
#
# # %% eRCNNSeqIter
# pred_variable = 'speed'
# pred_window = 3
# pred_detector = 'all_iter'
# pred_type = 'solo'
#
# out_seq = 3
# img_size = 20
#
# variables_list = ['flow', 'occupancy', 'speed']
# target = variables_list.index(pred_variable)
#
# result_folder = 'resultados/eRCNN/eRCNNSeqIter/ev1/error_mean_before/seq3/'
# model_name = f'best_observer.pt'
#
# if 'all' in pred_detector:
#     detectors_pred = 27
# else:
#     detectors_pred = 1
# out_size = 1 * detectors_pred
# hid_error_size = 6 * out_size
#
# model_type = 'eRCNNSeqIter'
# e_rcnn = eRCNNSeqIter(3, hid_error_size, out_size, pred_window=pred_window, out_seq=out_seq, dev=device)
# e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))
#
# day_periods = [0, 1, 2, 3]
# for day_period in day_periods:
#     test_set = STImgSeqDatasetDayTests(test_data_file_name, pred_detector=pred_detector,
#                                pred_window=pred_window, target=target, day_period=day_period)
#     test_set, _, _ = torch.utils.data.random_split(test_set, [12500, 12500, len(test_set) - 25000],
#                                                                generator=torch.Generator().manual_seed(seed))
#     eval_util.loss_evaluation(test_set,
#                               e_rcnn,
#                               model_type,
#                               batch_size=batch_size,
#                               res_folder=result_folder,
#                               file_sfx=f'_period_{day_period}',
#                               device=device,
#                               seed=seed)

# %% eREncDecSeq
pred_variable = 'speed'
pred_window = 3
pred_detector = 'all_iter'
pred_type = 'solo'

out_seq = pred_window  # Size of the out sequence
n_inputs_enc = 3  #nm
n_inputs_dec = 27  #nm
seqlen_rec = 6  # 8/12
hidden_size_rec = 40  # 7/20/40/50/70 --> best 50
num_layers_rec = 2   # 2/3/4

img_size = 20

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

result_folder = 'resultados/EncoderDecoder/eREncDecSeq/'
model_name = f'best_observer.pt'

if 'all' in pred_detector:
    detectors_pred = 27
else:
    detectors_pred = 1
out_size = 1 * detectors_pred

model_type = 'eREncDecSeq'
encod_decod = ErrorEncoderDecoder2D(n_inputs_enc=n_inputs_enc, n_inputs_dec=n_inputs_dec, n_outputs=out_size,
                               hidden_size=hidden_size_rec, num_layers=num_layers_rec, out_seq=out_seq,
                               error_size=seqlen_rec, dev=device)
encod_decod.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))

day_periods = [0, 1, 2, 3]
for day_period in day_periods:
    test_set = STImgSeqDatasetDayTests(test_data_file_name, pred_detector=pred_detector,
                               pred_window=pred_window, target=target, data_size=25000, day_period=day_period)
    test_set, _, _ = torch.utils.data.random_split(test_set, [12500, 12500, len(test_set) - 25000],
                                                               generator=torch.Generator().manual_seed(seed))
    eval_util.loss_evaluation(test_set,
                              encod_decod,
                              model_type,
                              batch_size=batch_size,
                              res_folder=result_folder,
                              file_sfx=f'_period_{day_period}',
                              device=device,
                              seed=seed)
