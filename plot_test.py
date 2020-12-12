import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from congestion_predict.data_sets import STImgSeqDataset
from congestion_predict.models import eRCNNSeq
from congestion_predict.utilities import count_parameters
import congestion_predict.plot as plt_util
import time
import json


# %% eRCNN
target = 2
label_conf = 'all'
out_seq = 4
img_size = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 50
batch_size = 1000  # only for the table

result_folder = 'resultados/eRCNN/new_model_sequence/seq4_detach/'
model_name = 'eRCNN_state_dict_model_2.pt'
# plt_util.plot_loss('MSE', f'loss_plot_train_{target}.txt', folder=result_folder)
# plt_util.plot_loss('MSE', f'mse_plot_valid_{target}.txt', target, folder=result_folder)
# plt_util.plot_loss('MAE', f'mae_plot_valid_{target}.txt', target, folder=result_folder)
# plt_util.plot_loss('MSE', f'mse_plot_test_{target}.txt', target, folder=result_folder)
# plt_util.plot_loss('MAE', f'mae_plot_test_{target}.txt', target, folder=result_folder)

if label_conf == 'all':
    detectors_pred = 27
else:
    detectors_pred = 1
hid_error_size = 6 * detectors_pred
out = 1 * detectors_pred
e_rcnn = eRCNNSeq(3, hid_error_size, out, out_seq=out_seq, dev=device)
e_rcnn.load_state_dict(torch.load(result_folder + model_name, map_location=torch.device(device)))
# plt_util.plot_image(target,
#                     label_conf,
#                     out_seq,
#                     img_size=img_size,
#                     model=e_rcnn,
#                     device=device,
#                     seed=seed,
#                     folder=result_folder)
#
# plt_util.plot_seq_out(target,
#                       label_conf,
#                       out_seq,
#                       model=e_rcnn,
#                       device=device,
#                       seed=seed,
#                       folder=result_folder)

plt_util.print_loss_table(target,
                          label_conf,
                          out_seq,
                          model=e_rcnn,
                          batch_size=batch_size,
                          device=device,
                          seed=seed)
