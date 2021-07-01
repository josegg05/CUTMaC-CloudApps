import numpy as np
import matplotlib.pyplot as plt
from congestion_predict.models import eRCNNSeq
from congestion_predict.utilities import count_parameters
from congestion_predict.data_sets import STImgSeqDataset
import torch
from torch import nn
import json
from prettytable import PrettyTable
import sys


def print_loss(loss_type, loss):
    if type(loss) == torch.Tensor:
        out = torch.mean(loss)
    else:
        out = np.mean(loss)
    print(f'{loss_type} = {out}\n')
    return out


def print_loss_sensor(loss_type, loss):
    # loss mus be calculated with no reduction
    tab = PrettyTable()
    if len(loss.shape) > 2:
        dim = (0, 1)
    else:
        dim = 0
    if type(loss) == torch.Tensor:
        out = torch.mean(loss, dim)
        tab.add_column("mean_detectors", out.numpy())
    else:
        out = np.mean(loss, dim)
        tab.add_column(f"{loss_type}_detectors", out)
    print(f'{loss_type} per sensor:')
    print(tab, '\n')
    return out


def print_loss_middle(loss_type, loss):
    # loss mus be calculated with no reduction
    if len(loss.shape) > 2:
        dim = (0, 1)
    else:
        dim = 0
    if type(loss) == torch.Tensor:
        out = torch.mean(loss, dim)
    else:
        out = np.mean(loss, dim)
    print(f'{loss_type} of the middle sensor = {out[int(loss.shape[-1]/2)]}\n')
    return out


def print_loss_seq(loss_type, loss):
    # loss mus be calculated with no reduction
    tab = PrettyTable()
    tab.field_names = [f"time_{i+1}" for i in range(loss.shape[1])]
    if type(loss) == torch.Tensor:
        out = torch.mean(loss, (0, 2))
        tab.add_row(out.numpy())
    else:
        out = np.mean(loss, (0, 2))
        tab.add_row(out)
    print(f'{loss_type} per sequence time-step:')
    print(tab, '\n')
    return out


def print_loss_sensor_seq(loss_type, loss):
    # loss mus be calculated with no reduction
    tab = PrettyTable()
    tab.field_names = [f"time_{i+1}" for i in range(loss.shape[1])]
    if type(loss) == torch.Tensor:
        out = torch.mean(loss, 0)
        tab.add_rows(out.transpose(0, 1).numpy())
    else:
        out = np.mean(loss, 0)
        tab.add_rows(out.transpose(0, 1))
    print(f'{loss_type} per sensor per sequence time-step')
    print(tab, '\n')
    return out


def print_loss_middle_seq(loss_type, loss):
    # loss mus be calculated with no reduction
    tab = PrettyTable()
    tab.field_names = [f"time_{i+1}" for i in range(loss.shape[1])]
    if type(loss) == torch.Tensor:
        out = torch.mean(loss, 0)
        tab.add_row(out.transpose(0, 1)[int(loss.shape[-1]/2)].numpy())
    else:
        out = np.mean(loss, 0)
        tab.add_row(out.transpose(0, 1)[int(loss.shape[-1]/2)])
    print(f'{loss_type} of the middle sensor per sequence time-step')
    print(tab, '\n')
    return out


def model_testing(test_set, model, model_type, batch_size, seqlen_rec, target_norm,  device,
                  stddev_torch=1, mean_torch=0, mod_lin=False):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model.to(device)  # Put the network on GPU if present
    criterion = nn.MSELoss(reduction='none')  # L2 Norm
    criterion2 = nn.L1Loss(reduction='none')
    model.eval()
    loss_mse_list = []
    loss_mae_list = []
    with torch.no_grad():
        for batch_idx, (inputs_test, targets_test) in enumerate(test_loader):
            if 'eRCNN'in model_type:
                inputs_test = inputs_test.transpose(0, 1)
                targets_test = targets_test.transpose(0, 1)
                inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

                outputs_test = model(inputs_test, targets_test)
                if target_norm:
                    outputs_test = outputs_test * stddev_torch + mean_torch

                if model_type == 'eRCNN':
                    loss_mse = criterion(outputs_test, targets_test[-1])
                    loss_mae = criterion2(outputs_test, targets_test[-1])
                elif model_type == 'eRCNNSeqLin':
                    out_seq = int(outputs_test.shape[1]/inputs_test.shape[3])
                    # print(outputs_test[0])
                    # print(outputs_test.view(batch_size, out_seq, -1)[0])
                    loss_mse = criterion(outputs_test.view(batch_size, out_seq, -1), targets_test[-1].view(batch_size, out_seq, -1))
                    loss_mae = criterion2(outputs_test.view(batch_size, out_seq, -1), targets_test[-1].view(batch_size, out_seq, -1))
                elif model_type == 'eRCNNSeq' or model_type == 'eRCNNSeqIter':
                    out_seq = outputs_test.shape[1]
                    loss_mse = criterion(outputs_test, targets_test[-out_seq:].transpose(0, 1))
                    loss_mae = criterion2(outputs_test, targets_test[-out_seq:].transpose(0, 1))

            elif model_type == 'EncoderDecoder':
                targets_test = torch.unsqueeze(targets_test, 1)
                X_c, X_r, Y = inputs_test.to(device), inputs_test[:, -seqlen_rec:, :].to(device), targets_test.to(device)

                outputs_test = model(X_c.float(), X_r.float())
                if target_norm:
                    outputs_test = outputs_test * stddev_torch + mean_torch

                loss_mse = criterion(outputs_test, Y)
                loss_mae = criterion2(outputs_test, Y)

            elif model_type == 'EncoderDecoder2D':
                targets_test = torch.unsqueeze(targets_test, 1)
                X_r_pre = inputs_test.transpose(1, 3)
                X_r_pre = X_r_pre.reshape(X_r_pre.shape[0], X_r_pre.shape[1], -1)
                X_c, X_r, Y = inputs_test.to(device), X_r_pre[:, -seqlen_rec:, :].to(device), targets_test.to(device)

                outputs_test = model(X_c.float(), X_r.float())
                if target_norm:
                    outputs_test = outputs_test * stddev_torch + mean_torch

                loss_mse = criterion(outputs_test, Y)
                loss_mae = criterion2(outputs_test, Y)

            elif model_type == 'eREncDecSeq':
                inputs_test = inputs_test.transpose(0, 1)
                targets_test = targets_test.transpose(0, 1)
                inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

                outputs_test = model(inputs_test, targets_test)
                if target_norm:
                    outputs_test = outputs_test * stddev_torch + mean_torch

                if mod_lin:
                    out_seq = int(outputs_test.shape[1]/targets_test.shape[-1])
                    print(outputs_test.view(batch_size, out_seq, -1).shape)
                    print(targets_test[-out_seq:].transpose(0, 1).shape)
                    loss_mse = criterion(outputs_test.view(batch_size, out_seq, -1), targets_test[-out_seq:].transpose(0, 1))
                    loss_mae = criterion2(outputs_test.view(batch_size, out_seq, -1), targets_test[-out_seq:].transpose(0, 1))

                else:
                    out_seq = outputs_test.shape[1]
                    print(f'preds: {outputs_test.shape}')
                    print(f'testy: {targets_test[-out_seq:].transpose(0, 1).shape}')
                    loss_mse = criterion(outputs_test, targets_test[-out_seq:].transpose(0, 1))
                    loss_mae = criterion2(outputs_test, targets_test[-out_seq:].transpose(0, 1))

            loss_mse_list.append(loss_mse)
            loss_mae_list.append(loss_mae)
            #break
    loss_mse = torch.cat(loss_mse_list, 0)
    loss_mae = torch.cat(loss_mae_list, 0)
    return loss_mse.cpu(), loss_mae.cpu()


def loss_evaluation(test_set, model, model_type, batch_size=0, seqlen_rec=12, res_folder='', file_sfx='',
                    target_norm=False, stddev_torch=1, mean_torch=0, mod_lin=False, device=None, seed=None, print_out=True):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(res_folder + f'loss_evaluation_{model_type}{file_sfx}.txt', 'w') as filehandle:
        sys.stdout = filehandle  # Change the standard output to the file we created.
        if seed is not None:
            torch.manual_seed(seed=seed)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
            # device = "cpu"

        if batch_size == 0:
            batch_size = len(test_set)

        count_parameters(model)

        loss_mse, loss_mae = model_testing(test_set, model, model_type, batch_size, seqlen_rec, target_norm, device,
                                           stddev_torch=stddev_torch, mean_torch=mean_torch, mod_lin=mod_lin)
        print(f'loss_mse: {loss_mse.shape}')
        print(f'loss_mae: {loss_mae.shape}')
        # loss_mae.shape --> (batch_size, seq_size, n_detect)
        print(' 1. ***********')
        print_loss('MSE', loss_mse)
        print(' 2. ***********')
        print_loss('MAE', loss_mae)
        print(' 3. ***********')
        print_loss_sensor('MAE', loss_mae)
        print(' 4. ***********')
        print_loss_middle('MAE', loss_mae)
        if 'Seq' in model_type:
            print(' 5. ***********')
            print_loss_seq('MAE', loss_mae)
            print(' 6. ***********')
            print_loss_sensor_seq('MAE', loss_mae)
            print(' 7. ***********')
            print_loss_middle_seq('MAE', loss_mae)

        sys.stdout = original_stdout  # Reset the standard output to its original value

    with open(res_folder + f'loss_evaluation_{model_type}{file_sfx}.txt', 'r') as filehandle:
        print(filehandle.read())


def plot_seq_out(target, pred_detector, out_seq, pred_window, model=None, model_path='', device=None, seed=None, folder=''):
    print('********************** Print a single sequence output **********************')
    if pred_detector == 'all':
        detectors_pred = 27
    else:
        detectors_pred = 1
    out_size = 1 * detectors_pred
    hid_error_size = 6 * out_size

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
        # device = "cpu"

    if model is None:
        e_rcnn = eRCNNSeq(3, hid_error_size, out_size, pred_window=pred_window, out_seq=out_seq, dev=device)
        e_rcnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    else:
        e_rcnn = model
    count_parameters(e_rcnn)
    val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
    val_test_set = STImgSeqDataset(val_test_data_file_name, pred_detector=pred_detector,
                           pred_window=pred_window, target=target, data_size=100000)

    e_rcnn.to(device)  # Put the network on GPU if present
    criterion = nn.MSELoss()  # L2 Norm
    criterion2 = nn.L1Loss()
    e_rcnn.eval()

    with torch.no_grad():
        if seed is not None:
            np.random.seed(seed=seed)
        inputs_test, targets_test = val_test_set[np.random.randint(99999)]
        inputs_test = inputs_test.unsqueeze_(1)
        targets_test = targets_test.unsqueeze_(1)
        inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

        outputs_test = e_rcnn(inputs_test, targets_test)

        loss = criterion(outputs_test, targets_test[-out_seq:].permute(1, 0, 2))
        loss2 = criterion2(outputs_test, targets_test[-out_seq:].permute(1, 0, 2))

    print(f"MSE = {loss}")
    print(f"MAE = {loss2}")

    tab_1 = PrettyTable()
    tab_2 = PrettyTable()
    tab_3 = PrettyTable()
    tab_1.field_names = ["time_1", "time_2", "time_3", "time_4"]
    tab_2.field_names = ["time_1", "time_2", "time_3", "time_4"]
    loss_sensors = torch.nn.functional.l1_loss(outputs_test, targets_test[-out_seq:].permute(1, 0, 2), reduction='none')[0].transpose(0, 1)
    tab_1.add_rows(loss_sensors.numpy())
    tab_2.add_row(torch.mean(loss_sensors, 0).numpy())
    tab_3.add_column("mean_detectors", torch.mean(loss_sensors, 1).numpy())
    print("MAE per detector per time-step")
    print(tab_1)
    print("MAE per time-step")
    print(tab_2)
    print("MAE per detector")
    print(tab_3)

    plt.figure()
    plt.title("Prediction image")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(outputs_test[0].transpose(1, 0).numpy())
    plt.savefig(folder + f"img_seq_out_{target}.png")
    plt.show()

    plt.figure()
    plt.title("Traget image")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(targets_test[-out_seq:].permute(1, 2, 0)[0].numpy())
    plt.savefig(folder + f"img_seq_target_{target}.png")
    plt.show()


def plot_image(target, pred_detector, out_seq, pred_window, pred_type='solo', model=None, model_path='', img_size=72, device=None, seed=None, folder='', shuffle=False):
    print('********************** Print a predicted image **********************')
    if seed is not None:
        torch.manual_seed(seed=seed)
    if pred_detector == 'all':
        detectors_pred = 27
    else:
        detectors_pred = 1
    out_size = 1 * detectors_pred
    hid_error_size = 6 * out_size

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
        # device = "cpu"

    if model is None:
        e_rcnn = eRCNNSeq(3, hid_error_size, out_size, pred_window=pred_window, out_seq=out_seq, dev=device)
        e_rcnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    else:
        e_rcnn = model
    count_parameters(e_rcnn)
    val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
    val_test_set = STImgSeqDataset(val_test_data_file_name, pred_detector=pred_detector,
                           pred_type=pred_type, pred_window=pred_window, target=target)
    if shuffle:
        val_test_set, test_set = torch.utils.data.random_split(val_test_set, [100000, 0])

    e_rcnn.to(device)  # Put the network on GPU if present
    criterion = nn.MSELoss()  # L2 Norm
    criterion2 = nn.L1Loss()
    e_rcnn.eval()

    losses_test_oneshot = []
    losses_test_oneshot2 = []
    outputs_test_oneshot = []
    targets_test_oneshot = []
    mae_each_sen = [[] for i in range(27)]
    if seed is not None:
        np.random.seed(seed=seed)
    first_idx = np.random.randint(99999-img_size)
    with torch.no_grad():
        for idx in range(img_size):
            inputs_test, targets_test = val_test_set[first_idx + idx]
            inputs_test = inputs_test.unsqueeze_(1)
            targets_test = targets_test.unsqueeze_(1)
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

            outputs_test = e_rcnn(inputs_test, targets_test)

            loss = criterion(outputs_test, targets_test[-out_seq:].permute(1, 0, 2))
            loss2 = criterion2(outputs_test, targets_test[-out_seq:].permute(1, 0, 2))
            losses_test_oneshot.append(loss.item())
            losses_test_oneshot2.append(loss2.item())

            outputs_test_oneshot.append(outputs_test[-1, -1:])
            targets_test_oneshot.append(targets_test[-out_seq:].permute(1, 0, 2)[-1, -1:])

            for j in range(27):
                mae_each_sen[j].append(criterion2(outputs_test[0][-1][j], targets_test[-out_seq:].permute(1, 0, 2)[0][-1][j]))

    print(f"MSE = {np.mean(losses_test_oneshot)}")
    print(f"MAE = {np.mean(losses_test_oneshot2)}")

    mae_mean_each_sen = [0 for i in range(27)]
    for i in range(27):
        mae_mean_each_sen[i] = np.mean(mae_each_sen[i])
    print(mae_mean_each_sen)

    outputs_test_oneshot = torch.cat(outputs_test_oneshot)
    targets_test_oneshot = torch.cat(targets_test_oneshot)
    outputs_test_oneshot = outputs_test_oneshot.permute(1, 0)
    targets_test_oneshot = targets_test_oneshot.permute(1, 0)

    plt.figure()
    plt.title("Prediction image")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(outputs_test_oneshot.cpu())
    plt.savefig(folder + f"img_out_{target}.png")
    plt.show()

    plt.figure()
    plt.title("Target image")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(targets_test_oneshot.cpu())
    plt.savefig(folder + f"img_target_{target}.png")
    plt.show()

    # print the diff
    diff1 = abs(outputs_test_oneshot - targets_test_oneshot)
    diff1 = diff1 / diff1.max()

    plt.figure()
    plt.title("Differences between images - Target")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(torch.ones(diff1.shape) - diff1.cpu(), cmap="gray")
    plt.savefig(folder + f"diff_{target}.png")
    plt.show()

    # plt.figure()
    # plt.plot(losses_test_oneshot)
    # plt.title("Training MSE")
    # plt.ylabel("MSE")
    # plt.xlabel("Bacthx100")
    # plt.grid()
    # plt.savefig(folder + f"train_mse_oneshot_{target}.png")
    # plt.show()
    #
    # plt.figure()
    # plt.plot(losses_test_oneshot2)
    # # plt.ylim(0, 3)
    # plt.title("Testing MAE")
    # plt.ylabel("MAE")
    # plt.xlabel("Bacth")
    # plt.grid()
    # plt.savefig(folder + f"test_mae_oneshot_{target}.png")
    # plt.show()


def print_loss_table(target, pred_detector, out_seq, pred_window, pred_type='solo', model=None, model_path='', batch_size=100, device=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed=seed)
    if pred_detector == 'all':
        detectors_pred = 27
    else:
        detectors_pred = 1
    out_size = 1 * detectors_pred
    hid_error_size = 6 * out_size

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
        # device = "cpu"

    if model is None:
        e_rcnn = eRCNNSeq(3, hid_error_size, out_size, pred_window=pred_window, out_seq=out_seq, dev=device)
        e_rcnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    else:
        e_rcnn = model

    count_parameters(e_rcnn)
    val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
    val_test_set = STImgSeqDataset(val_test_data_file_name, pred_detector=pred_detector,
                           pred_type=pred_type, pred_window=pred_window, target=target)
    val_test_loader = torch.utils.data.DataLoader(val_test_set, batch_size=batch_size, shuffle=True)

    e_rcnn.to(device)  # Put the network on GPU if present
    criterion = nn.MSELoss(reduction='none')  # L2 Norm
    criterion2 = nn.L1Loss(reduction='none')
    e_rcnn.eval()
    with torch.no_grad():
        for batch_idx, (inputs_test, targets_test) in enumerate(val_test_loader):
            inputs_test = inputs_test.transpose(0, 1)
            targets_test = targets_test.transpose(0, 1)
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)

            outputs_test = e_rcnn(inputs_test, targets_test)

            loss = criterion(outputs_test, targets_test[-out_seq:].transpose(0, 1))
            loss2 = criterion2(outputs_test, targets_test[-out_seq:].transpose(0, 1))
            break

    loss2 = torch.mean(loss2, 0).transpose(0, 1)
    tab_1 = PrettyTable()
    tab_2 = PrettyTable()
    tab_3 = PrettyTable()
    tab_1.field_names = ["time_1", "time_2", "time_3", "time_4"]
    tab_2.field_names = ["time_1", "time_2", "time_3", "time_4"]
    tab_1.add_rows(loss2.numpy())
    tab_2.add_row(torch.mean(loss2, 0).numpy())
    tab_3.add_column("mean_detectors", torch.mean(loss2, 1).numpy())
    print("MAE per detector per time-step")
    print(tab_1)
    print("MAE per time-step")
    print(tab_2)
    print("MAE per detector")
    print(tab_3)

    loss = torch.mean(loss, 0).transpose(0, 1)
    tab_1 = PrettyTable()
    tab_2 = PrettyTable()
    tab_3 = PrettyTable()
    tab_1.field_names = ["time_1", "time_2", "time_3", "time_4"]
    tab_2.field_names = ["time_1", "time_2", "time_3", "time_4"]
    tab_1.add_rows(loss.numpy())
    tab_2.add_row(torch.mean(loss, 0).numpy())
    tab_3.add_column("mean_detectors", torch.mean(loss, 1).numpy())
    print("MSE per detector per time-step")
    print(tab_1)
    print("MSE per time-step")
    print(tab_2)
    print("MSE per detector")
    print(tab_3)


'''Training script Plot and Prints functions'''
def plot_loss(loss_name, loss_filename, tt='Test', folder=''):
    with open(folder + loss_filename, 'r') as filehandle:
        loss_plot = json.load(filehandle)

    flatList = [item for elem in loss_plot for item in elem]
    plt.figure()
    plt.plot(flatList)
    #plt.ylim(0, 3)
    plt.title(f"{tt}ing {loss_name}")
    plt.ylabel(loss_name)
    plt.xlabel("Bacth")
    plt.grid()
    plt.savefig(folder + f"{loss_filename}.png")

    flatList_mean = [np.mean(elem) for elem in loss_plot]
    plt.figure()
    plt.plot(flatList_mean)
    #plt.ylim(0, 2)
    plt.title(f"{tt}ing {loss_name}")
    plt.ylabel(loss_name)
    plt.xlabel("Bacthx100")
    plt.grid()
    plt.savefig(folder + f"{loss_filename}_mean.png")

    plt.show()


def plot_mse(loss_mse_filename, target, folder=''):
    # loss_plot_test = []
    # with open(folder + loss_mse_filename, 'r') as filehandle:
    #     for line in filehandle:
    #         # remove linebreak which is the last character of the string
    #         currentPlace = ast.literal_eval(line[:-1])
    #
    #         # add item to the list
    #         loss_plot_test.append(currentPlace)
    with open(folder + loss_mse_filename, 'r') as filehandle:
        loss_plot_test = json.load(filehandle)

    flatList_test1 = [item for elem in loss_plot_test for item in elem]
    plt.figure()
    plt.plot(flatList_test1)
    #plt.ylim(0, 40)
    plt.title("Testing MSE")
    plt.ylabel("MSE")
    plt.xlabel("Bacth")
    plt.grid()
    plt.savefig(folder + f"test_mse_{target}.png")

    flatList_mean_test1 = [np.mean(elem) for elem in loss_plot_test]
    plt.figure()
    plt.plot(flatList_mean_test1)
    #plt.ylim(0, 10)
    plt.title("Testing MSE")
    plt.ylabel("MSE")
    plt.xlabel("Bacthx100")
    plt.grid()
    plt.savefig(folder + f"test_mean_mse_{target}.png")

    plt.show()


def plot_mae(loss_mae_filename, target, folder=''):
    # loss_plot_test2 = []
    # with open(folder + loss_mae_filename, 'r') as filehandle:
    #     for line in filehandle:
    #         # remove linebreak which is the last character of the string
    #         currentPlace = ast.literal_eval(line[:-1])
    #
    #         # add item to the list
    #         loss_plot_test2.append(currentPlace)
    with open(folder + loss_mae_filename, 'r') as filehandle:
        loss_plot_test2 = json.load(filehandle)

    flatList_test2 = [item for elem in loss_plot_test2 for item in elem]
    plt.figure()
    plt.plot(flatList_test2)
    #plt.ylim(0, 3)
    plt.title("Testing MAE")
    plt.ylabel("MAE")
    plt.xlabel("Bacth")
    plt.grid()
    plt.savefig(folder + f"test_mae_{target}.png")

    flatList_mean_test2 = [np.mean(elem) for elem in loss_plot_test2]
    plt.figure()
    plt.plot(flatList_mean_test2)
    #plt.ylim(0, 2)
    plt.title("Testing MAE")
    plt.ylabel("MAE")
    plt.xlabel("Bacthx100")
    plt.grid()
    plt.savefig(folder + f"test_mean_mae_{target}.png")

    plt.show()