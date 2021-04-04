import torch
from congestion_predict.data_sets import STImgSeqDataset, STImgSeqDatasetMTER_LA
from congestion_predict.models import eRCNNSeq, eRCNNSeqLin, eRCNNSeqIter, eRCNN, ErrorEncoderDecoder2D
import congestion_predict.evaluation as eval_util
import numpy as np


# global variables initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 50
batch_size = 1000
test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"

# %% eRCNN
pred_variable = 'speed'
pred_window = 3
pred_detector = 'all'
pred_type = 'solo'

img_size = 20

result_folder = 'resultados/eRCNN/eRCNN/ev1/all_solo_15_ev1/'
model_name = 'best_observer.pt'

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

data_size=100000
test_set = STImgSeqDataset(test_data_file_name, pred_detector=pred_detector,
                           pred_type=pred_type, pred_window=pred_window, target=target, data_size=100000)
test_set, _, _ = torch.utils.data.random_split(test_set, [50000, 50000, len(test_set) - 100000],
                                                           generator=torch.Generator().manual_seed(seed))

if pred_detector == 'all':
    detectors_pred = 27
else:
    detectors_pred = 1
out_size = 1 * detectors_pred
hid_error_size = 6 * out_size

model_type = 'eRCNN'
e_rcnn = eRCNN(3, hid_error_size, out_size, pred_window=pred_window, dev=device)
e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))

eval_util.loss_evaluation(test_set,
                          e_rcnn,
                          model_type,
                          batch_size=batch_size,
                          res_folder=result_folder,
                          device=device,
                          seed=seed)

# %% eRCNNSeq
pred_variable = 'speed'
pred_window = 3
pred_detector = 'all'
pred_type = 'solo'

out_seq = 3
img_size = 20

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

result_folder = 'resultados/eRCNN/eRCNNSeq/seq4_no_detach/'
model_name = f'eRCNN_state_dict_model_{target}.pt'

data_size=100000
test_set = STImgSeqDataset(test_data_file_name, pred_detector=pred_detector,
                           pred_type=pred_type, pred_window=pred_window, target=target, data_size=100000)
test_set, _, _ = torch.utils.data.random_split(test_set, [50000, 50000, len(test_set) - 100000],
                                                           generator=torch.Generator().manual_seed(seed))

if pred_detector == 'all':
    detectors_pred = 27
else:
    detectors_pred = 1
out_size = 1 * detectors_pred
hid_error_size = 6 * out_size

model_type = 'eRCNNSeq'
e_rcnn = eRCNNSeq(3, hid_error_size, out_size, pred_window=pred_window, out_seq=out_seq, dev=device)
e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))

eval_util.loss_evaluation(test_set,
                          e_rcnn,
                          model_type,
                          batch_size=batch_size,
                          res_folder=result_folder,
                          device=device,
                          seed=seed)


# eval_util.plot_image(target,
#                     pred_detector,
#                     out_seq,
#                     img_size=img_size,
#                     model=e_rcnn,
#                     device=device,
#                     seed=seed,
#                     folder=result_folder)
#
# eval_util.plot_seq_out(target,
#                       pred_detector,
#                       out_seq,
#                       model=e_rcnn,
#                       device=device,
#                       seed=seed,
#                       folder=result_folder)
#
# eval_util.print_loss_table(target,
#                           pred_detector,
#                           out_seq,
#                           model=e_rcnn,
#                           batch_size=batch_size,
#                           device=device,
#                           seed=seed)
#
# eval_util.plot_loss('MSE', f'loss_plot_train_{target}.txt', folder=result_folder)
# eval_util.plot_loss('MSE', f'mse_plot_valid_{target}.txt', target, folder=result_folder)
# eval_util.plot_loss('MAE', f'mae_plot_valid_{target}.txt', target, folder=result_folder)
# eval_util.plot_loss('MSE', f'mse_plot_test_{target}.txt', target, folder=result_folder)
# eval_util.plot_loss('MAE', f'mae_plot_test_{target}.txt', target, folder=result_folder)


# %% eRCNNSeqLin
dataset = 'metr_la'  # cali_i5, metr_la, vegas_i15
pred_variable = 'speed'
pred_window = 12
pred_detector = 'all_lin'
pred_type = 'solo'
seq_size = 72
image_size = 72  # for the cali_i5 dataset
target_norm = False


extra_fc = []
out_seq = 3
img_size = 20

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

result_folder = 'resultados/eRCNN/eRCNNSeqLin/'
model_name = f'ev1/[256]/eRCNN_state_dict_model_{target}.pt'

if dataset == 'cali_i5':
    test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
    test_set = STImgSeqDataset(test_data_file_name, pred_detector=pred_detector,
                               pred_type=pred_type, pred_window=pred_window, target=target,
                               seq_size=seq_size, image_size=image_size, target_norm=target_norm,
                               data_size=100000)

    test_set, _, _ = torch.utils.data.random_split(test_set, [50000, 50000, len(test_set) - 100000],
                                                   generator=torch.Generator().manual_seed(seed))
elif dataset == 'metr_la':
    test_data_file_name = 'datasets/METR-LA/test_filtered_we.npz'
    test_data_temp = np.load(test_data_file_name)
    test_data = {'x': test_data_temp['x'], 'y': test_data_temp['y']}
    test_data_temp.close()

    test_set = STImgSeqDatasetMTER_LA(test_data, pred_detector=pred_detector, seq_size=seq_size,
                                      pred_type=pred_type, pred_window=pred_window, target=target)

image_seq, label = test_set[0]

if 'all' in pred_detector:
    detect_num = image_seq.shape[2]
else:
    detect_num = 1
image_size = image_seq.shape[-1]
out_size = detect_num * pred_window
hid_error_size = 6 * out_size

model_type = 'eRCNNSeqLin'
e_rcnn = eRCNNSeqLin(image_seq.shape[1], detect_num, image_size, out_size, pred_window, fc_pre_outs=extra_fc, dev=device)
e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))

eval_util.loss_evaluation(test_set,
                          e_rcnn,
                          model_type,
                          batch_size=batch_size,
                          res_folder=result_folder,
                          device=device,
                          seed=seed)

# %% eRCNNSeqIter
dataset = 'metr_la'  # cali_i5, metr_la, vegas_i15
pred_variable = 'speed'
pred_window = 12
pred_detector = 'all_iter'
pred_type = 'solo'
seq_size = 12
image_size = 72  # for the cali_i5 dataset
target_norm = False

out_seq = pred_window
img_size = 20

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

result_folder = 'resultados/eRCNN/eRCNNSeqIter/'
model_name = f'ev1_good/mter_la/seq12_pw12/best_observer.pt'

if dataset == 'cali_i5':
    test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
    test_set = STImgSeqDataset(test_data_file_name, pred_detector=pred_detector,
                               pred_type=pred_type, pred_window=pred_window, target=target,
                               seq_size=seq_size, image_size=image_size, target_norm=target_norm,
                               data_size=100000)

    test_set, _, _ = torch.utils.data.random_split(test_set, [50000, 50000, len(test_set) - 100000],
                                                   generator=torch.Generator().manual_seed(seed))
elif dataset == 'metr_la':
    test_data_file_name = 'datasets/METR-LA/test_filtered_we.npz'
    test_data_temp = np.load(test_data_file_name)
    test_data = {'x': test_data_temp['x'], 'y': test_data_temp['y']}
    test_data_temp.close()

    test_set = STImgSeqDatasetMTER_LA(test_data, pred_detector=pred_detector, seq_size=seq_size,
                                      pred_type=pred_type, pred_window=pred_window, target=target)

image_seq, label = test_set[0]

if 'all' in pred_detector:
    detect_num = image_seq.shape[-2]
else:
    detect_num = 1
image_size = image_seq.shape[-1]
out_size = 1 * detect_num
# hid_error_size = 6 * out_size

model_type = 'eRCNNSeqIter'
e_rcnn = eRCNNSeqIter(image_seq.shape[1], detect_num, image_size, out_size, pred_window=pred_window, out_seq=out_seq, dev=device)
e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))

eval_util.loss_evaluation(test_set,
                          e_rcnn,
                          model_type,
                          batch_size=batch_size,
                          res_folder=result_folder,
                          device=device,
                          seed=seed)

# %% eREncDecSeq
dataset = 'metr_la'  # cali_i5, metr_la, vegas_i15
pred_variable = 'speed'
pred_window = 12
pred_detector = 'all_iter'
pred_type = 'solo'
seq_size = 24
image_size = 72  # for the cali_i5 dataset
target_norm = False

out_seq = pred_window  # Size of the out sequence
n_inputs_enc = 3  # nm
n_inputs_dec = 27  # nm
seqlen_rec = 6  # 8/12
hidden_size_rec = 40  # 7/20/40/50/70 --> best 50
num_layers_rec = 2  # 2/3/4

img_size = 20

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

result_folder = 'resultados/EncoderDecoder/eREncDecSeq/'
model_name = f'metr_la/seq_12_predw_12/best_observer.pt'

if dataset == 'cali_i5':
    test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
    test_set = STImgSeqDataset(test_data_file_name, pred_detector=pred_detector,
                               pred_type=pred_type, pred_window=pred_window, target=target,
                               seq_size=seq_size, image_size=image_size, target_norm=target_norm,
                               data_size=100000)

    test_set, _, _ = torch.utils.data.random_split(test_set, [50000, 50000, len(test_set) - 100000],
                                                   generator=torch.Generator().manual_seed(seed))
elif dataset == 'metr_la':
    test_data_file_name = 'datasets/METR-LA/test_filtered_we.npz'
    test_data_temp = np.load(test_data_file_name)
    test_data = {'x': test_data_temp['x'], 'y': test_data_temp['y']}
    test_data_temp.close()

    test_set = STImgSeqDatasetMTER_LA(test_data, pred_detector=pred_detector, seq_size=seq_size,
                                      pred_type=pred_type, pred_window=pred_window, target=target)

image_seq, label = test_set[0]

if 'all' in pred_detector:
    detect_num = image_seq.shape[-2]
else:
    detect_num = 1
image_size = image_seq.shape[-1]
out_size = 1 * detect_num

model_type = 'eREncDecSeq'
encod_decod = ErrorEncoderDecoder2D(n_inputs_enc=image_seq.shape[1], n_inputs_dec=detect_num, n_outputs=out_size,
                                    hidden_size=hidden_size_rec, num_layers=num_layers_rec, out_seq=out_seq,
                                    error_size=seqlen_rec, image_size=image_size, dev=device)
encod_decod.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))

eval_util.loss_evaluation(test_set,
                          encod_decod,
                          model_type,
                          batch_size=batch_size,
                          res_folder=result_folder,
                          device=device,
                          seed=seed)
