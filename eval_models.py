import torch
from congestion_predict.data_sets import STImgSeqDataset
from congestion_predict.models import eRCNNSeq, eRCNNSeqLin, eRCNNSeqIter
import congestion_predict.evaluation as eval_util


# global variables initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 50
batch_size = 1000
test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"

# %% eRCNNSeq
pred_variable = 'speed'
pred_window = 4
pred_detector = 'all'
pred_type = 'solo'

out_seq = 4
img_size = 20

result_folder = 'resultados/eRCNN/eRCNNSeq/seq4_no_detach/'
model_name = 'eRCNN_state_dict_model_2.pt'

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
pred_variable = 'speed'
pred_window = 4
pred_detector = 'all_lin'
pred_type = 'solo'

out_seq = 4
img_size = 20

result_folder = 'resultados/eRCNN/eRCNNSeqLin/best_fc_256/'
model_name = f'eRCNN_state_dict_model_[256]_{target}.pt'

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

test_set = STImgSeqDataset(test_data_file_name, pred_detector=pred_detector,
                           pred_type=pred_type, pred_window=pred_window, target=target, data_size=100000)
test_set, _, _ = torch.utils.data.random_split(test_set, [50000, 50000, len(test_set) - 100000],
                                                           generator=torch.Generator().manual_seed(seed))

if 'all' in pred_detector:
    detectors_pred = 27
else:
    detectors_pred = 1
out_size = detectors_pred * pred_window
hid_error_size = 6 * out_size
model_type = 'eRCNNSeqLin'
e_rcnn = eRCNNSeqLin(3, hid_error_size, out_size, pred_window, fc_pre_outs=[256], dev=device)
e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))

eval_util.loss_evaluation(test_set,
                          e_rcnn,
                          model_type,
                          batch_size=batch_size,
                          res_folder=result_folder,
                          device=device,
                          seed=seed)

# %% eRCNNSeqIter
pred_variable = 'speed'
pred_window = 4
pred_detector = 'all_iter'
pred_type = 'solo'

out_seq = 4
img_size = 20

result_folder = 'resultados/eRCNN/eRCNNSeqIter/seq_4/'
model_name = f'best_observer.pt'

variables_list = ['flow', 'occupancy', 'speed']
target = variables_list.index(pred_variable)

test_set = STImgSeqDataset(test_data_file_name, pred_detector=pred_detector,
                           pred_type=pred_type, pred_window=pred_window, target=target, data_size=100000)
test_set, _, _ = torch.utils.data.random_split(test_set, [50000, 50000, len(test_set) - 100000],
                                                           generator=torch.Generator().manual_seed(seed))

if 'all' in pred_detector:
    detectors_pred = 27
else:
    detectors_pred = 1
out_size = 1 * detectors_pred
hid_error_size = 6 * out_size

model_type = 'eRCNNSeqIter'
e_rcnn = eRCNNSeqIter(3, hid_error_size, out_size, pred_window=pred_window, out_seq=out_seq, dev=device)
e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))

eval_util.loss_evaluation(test_set,
                          e_rcnn,
                          model_type,
                          batch_size=batch_size,
                          res_folder=result_folder,
                          device=device,
                          seed=seed)
