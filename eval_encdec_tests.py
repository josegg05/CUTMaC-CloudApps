import torch
from congestion_predict.data_sets import STImgSeqDataset, STImgSeqDatasetMTER_LA
from congestion_predict.models import eRCNNSeqLin, ErrorEncoderDecoder2D, ErrorEncoderDecoder2DLin
import congestion_predict.evaluation as eval_util
import numpy as np


# global variables initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 50
batch_size = 1000

# %% eREncDecSeq
dataset = 'metr_la'  # cali_i5, metr_la, vegas_i15
pred_variable = 'speed'
pred_window = 12
pred_detector = 'all_iter'
pred_type = 'solo'
seq_size = 12
image_size = 72  # for the cali_i5 dataset
target_norm = False

out_seq = pred_window  # Size of the out sequence
seqlen_rec = 6  # 8/12
hidden_size_rec = 40  # 7/20/40/50/70 --> best 50
num_layers_rec = 2  # 2/3/4

img_size = 20

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

result_folder = 'resultados/EncoderDecoder/'
model_name_list = [f'Results/main_tests/3_error_all_zeros/best_observer.pt',
                   f'Results/main_tests/5_train_mean_stddev/best_observer.pt',
                   f'Results/main_tests/6_norm/best_observer.pt',
                   f'Results/main_tests/7_error_lin/best_observer.pt',
                   f'Results/main_tests/8_errorlin_norm/best_observer.pt']
sufix_list =['3_error_all_zeros',
             '5_train_mean_stddev',
             '6_norm',
             '7_error_lin',
             '8_errorlin_norm']

if dataset == 'cali_i5':
    test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
    test_set = STImgSeqDataset(test_data_file_name, pred_detector=pred_detector,
                               pred_type=pred_type, pred_window=pred_window, target=target,
                               seq_size=seq_size, image_size=image_size, target_norm=target_norm,
                               data_size=100000)

    test_set, _, _ = torch.utils.data.random_split(test_set, [50000, 50000, len(test_set) - 100000],
                                                   generator=torch.Generator().manual_seed(seed))
elif dataset == 'metr_la':
    train_data_file_name = 'datasets/METR-LA/train_filtered_we.npz'
    train_data_temp = np.load(train_data_file_name)
    train_data = {'x': train_data_temp['x'], 'y': train_data_temp['y']}
    mean = train_data['x'][..., 0].mean()
    stddev = train_data['x'][..., 0].std()
    train_data_temp.close()

    test_data_file_name = 'datasets/METR-LA/test_filtered_we.npz'
    test_data_temp = np.load(test_data_file_name)
    test_data = {'x': test_data_temp['x'], 'y': test_data_temp['y']}
    test_data_temp.close()

    test_set = STImgSeqDatasetMTER_LA(test_data, mean=mean, stddev=stddev, pred_detector=pred_detector,
                                      seq_size=seq_size,
                                      pred_type=pred_type, pred_window=pred_window, target=target,
                                      target_norm=target_norm)
    stddev_torch = torch.Tensor([stddev]).to(device)
    mean_torch = torch.Tensor([mean]).to(device)

image_seq, label = test_set[0]

if 'all' in pred_detector:
    detect_num = image_seq.shape[-2]
else:
    detect_num = 1
image_size = image_seq.shape[-1]
out_size = 1 * detect_num

model_type = 'eREncDecSeq'
batch_size = len(test_set)
print(f'batch_size = {batch_size}')
for idx in range(len(model_name_list)):
    model_name = model_name_list[idx]
    sufix = sufix_list[idx]
    if idx == 0:
        all_zeros = True
    else:
        all_zeros = False

    if idx == 2 or idx == 4:
        target_norm = True
    else:
        target_norm = False

    if idx == 3 or idx == 4:
        mod_lin = True
    else:
        mod_lin = False

    if mod_lin:
        encod_decod = ErrorEncoderDecoder2DLin(n_inputs_enc=image_seq.shape[1], n_inputs_dec=detect_num,
                                               n_outputs=out_size,
                                               hidden_size=hidden_size_rec, num_layers=num_layers_rec, out_seq=out_seq,
                                               error_size=seqlen_rec, image_size=image_size, dev=device)

    else:
        encod_decod = ErrorEncoderDecoder2D(n_inputs_enc=image_seq.shape[1], n_inputs_dec=detect_num, n_outputs=out_size,
                                            hidden_size=hidden_size_rec, num_layers=num_layers_rec, out_seq=out_seq,
                                            error_size=seqlen_rec, image_size=image_size, all_zeros=all_zeros, dev=device)
    encod_decod.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))

    eval_util.loss_evaluation(test_set,
                              encod_decod,
                              model_type,
                              batch_size=batch_size,
                              res_folder=result_folder,
                              file_sfx=sufix,
                              target_norm=target_norm,
                              stddev_torch=stddev_torch,
                              mean_torch=mean_torch,
                              mod_lin=mod_lin,
                              device=device,
                              seed=seed)


# # %% eRCNNSeqLin
# dataset = 'metr_la'  # cali_i5, metr_la, vegas_i15
# pred_variable = 'speed'
# pred_window = 12
# pred_detector = 'all_lin'
# pred_type = 'solo'
# seq_size = 72
# image_size = 72  # for the cali_i5 dataset
# target_norm = False
#
#
# extra_fc = []
# out_seq = 3
# img_size = 20
#
# variables_list = ['flow', 'occupancy', 'speed']
# target = variables_list.index(pred_variable)
#
# result_folder = 'resultados/eRCNN/eRCNNSeqLin/'
# model_name = f'ev1/[256]/eRCNN_state_dict_model_{target}.pt'
#
# if dataset == 'cali_i5':
#     test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
#     test_set = STImgSeqDataset(test_data_file_name, pred_detector=pred_detector,
#                                pred_type=pred_type, pred_window=pred_window, target=target,
#                                seq_size=seq_size, image_size=image_size, target_norm=target_norm,
#                                data_size=100000)
#
#     test_set, _, _ = torch.utils.data.random_split(test_set, [50000, 50000, len(test_set) - 100000],
#                                                    generator=torch.Generator().manual_seed(seed))
# elif dataset == 'metr_la':
#     test_data_file_name = 'datasets/METR-LA/test_filtered_we.npz'
#     test_data_temp = np.load(test_data_file_name)
#     test_data = {'x': test_data_temp['x'], 'y': test_data_temp['y']}
#     test_data_temp.close()
#
#     test_set = STImgSeqDatasetMTER_LA(test_data, pred_detector=pred_detector, seq_size=seq_size,
#                                       pred_type=pred_type, pred_window=pred_window, target=target)
#
# image_seq, label = test_set[0]
#
# if 'all' in pred_detector:
#     detect_num = image_seq.shape[2]
# else:
#     detect_num = 1
# image_size = image_seq.shape[-1]
# out_size = detect_num * pred_window
# hid_error_size = 6 * out_size
#
# model_type = 'eRCNNSeqLin'
# e_rcnn = eRCNNSeqLin(image_seq.shape[1], detect_num, image_size, out_size, pred_window, fc_pre_outs=extra_fc, dev=device)
# e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))
#
# eval_util.loss_evaluation(test_set,
#                           e_rcnn,
#                           model_type,
#                           batch_size=batch_size,
#                           res_folder=result_folder,
#                           device=device,
#                           seed=seed)
#
