import numpy as np
import matplotlib.pyplot as plt
from congestion_predict.models import eRCNNSeq
from congestion_predict.utilities import count_parameters
from congestion_predict.data_sets import STImgSeqDataset
import torch
from torch import nn
import ast


def plot_mse(loss_mse_filename, target, folder=''):
    loss_plot_test = []
    with open(folder + loss_mse_filename, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = ast.literal_eval(line[:-1])

            # add item to the list
            loss_plot_test.append(currentPlace)

    flatList_test1 = [item for elem in loss_plot_test for item in elem]
    plt.figure(1)
    plt.plot(flatList_test1)
    plt.ylim(0, 40)
    plt.title("Testing MSE")
    plt.ylabel("MSE")
    plt.xlabel("Bacth")
    plt.grid()
    plt.savefig(folder + f"test_mse_{target}.png")

    flatList_mean_test1 = [np.mean(elem) for elem in loss_plot_test]
    plt.figure(3)
    plt.plot(flatList_mean_test1)
    plt.ylim(0, 10)
    plt.title("Testing MSE")
    plt.ylabel("MSE")
    plt.xlabel("Bacthx100")
    plt.grid()
    plt.savefig(folder + f"test_mean_mse_{target}.png")

    plt.show()


def plot_mae(loss_mae_filename, target, folder=''):
    loss_plot_test2 = []
    with open(folder + loss_mae_filename, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = ast.literal_eval(line[:-1])

            # add item to the list
            loss_plot_test2.append(currentPlace)

    flatList_test2 = [item for elem in loss_plot_test2 for item in elem]
    plt.figure(2)
    plt.plot(flatList_test2)
    plt.ylim(0, 3)
    plt.title("Testing MAE")
    plt.ylabel("MAE")
    plt.xlabel("Bacth")
    plt.grid()
    plt.savefig(folder + f"test_mae_{target}.png")

    flatList_mean_test2 = [np.mean(elem) for elem in loss_plot_test2]
    plt.figure(4)
    plt.plot(flatList_mean_test2)
    plt.ylim(0, 2)
    plt.title("Testing MAE")
    plt.ylabel("MAE")
    plt.xlabel("Bacthx100")
    plt.grid()
    plt.savefig(folder + f"test_mean_mae_{target}.png")

    plt.show()

def plot_loss(loss_name, loss_filename, tt='Test', folder=''):
    loss_plot = []
    with open(folder + loss_filename, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = ast.literal_eval(line[:-1])

            # add item to the list
            loss_plot.append(currentPlace)

    flatList = [item for elem in loss_plot for item in elem]
    plt.figure(2)
    plt.plot(flatList)
    plt.ylim(0, 3)
    plt.title(f"{tt}ing {loss_name}")
    plt.ylabel(loss_name)
    plt.xlabel("Bacth")
    plt.grid()
    plt.savefig(folder + f"{loss_filename}.png")

    flatList_mean = [np.mean(elem) for elem in loss_plot]
    plt.figure(4)
    plt.plot(flatList_mean)
    plt.ylim(0, 2)
    plt.title(f"{tt}ing {loss_name}")
    plt.ylabel(loss_name)
    plt.xlabel("Bacthx100")
    plt.grid()
    plt.savefig(folder + f"{loss_filename}_mean.png")

    plt.show()


def plot_seq_out(target, label_conf, out_seq, model=None, model_path='', device=None, seed=None, folder=''):
    if label_conf == 'all':
        detectors_pred = 27
    else:
        detectors_pred = 1
    hid_error_size = 6 * detectors_pred
    out = 1 * detectors_pred
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
        # device = "cpu"

    if model is None:
        e_rcnn = eRCNNSeq(3, hid_error_size, out, out_seq=out_seq, dev=device)
        e_rcnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    else:
        e_rcnn = model
    count_parameters(e_rcnn)
    val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
    val_test_set = STImgSeqDataset(val_test_data_file_name, data_size=100000, label_conf=label_conf, target=target)

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

    plt.figure(1)
    plt.title("Prediction image")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(outputs_test[0].transpose(1, 0).numpy())
    plt.savefig(folder + f"img_seq_out_{target}.png")
    plt.show()

    plt.figure(2)
    plt.title("Traget image")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(targets_test[-out_seq:].permute(1, 2, 0)[0].numpy())
    plt.savefig(folder + f"img_seq_target_{target}.png")
    plt.show()


def plot_image(target, label_conf, out_seq, model=None, model_path='', img_size=72, device=None, seed=None, folder=''):
    if label_conf == 'all':
        detectors_pred = 27
    else:
        detectors_pred = 1
    hid_error_size = 6 * detectors_pred
    out = 1 * detectors_pred
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check whether a GPU is present.
        # device = "cpu"

    if model is None:
        e_rcnn = eRCNNSeq(3, hid_error_size, out, out_seq=out_seq, dev=device)
        e_rcnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    else:
        e_rcnn = model
    count_parameters(e_rcnn)
    val_test_data_file_name = "datasets/california_paper_eRCNN/I5-N-3/2016.csv"
    val_test_set = STImgSeqDataset(val_test_data_file_name, data_size=100000, label_conf=label_conf, target=target)

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

    print(f"MAE pre = {np.mean(losses_test_oneshot)}")
    print(f"MAE = {np.mean(losses_test_oneshot2)}")

    mae_mean_each_sen = [0 for i in range(27)]
    for i in range(27):
        mae_mean_each_sen[i] = np.mean(mae_each_sen[i])
    print(mae_mean_each_sen)

    outputs_test_oneshot = torch.cat(outputs_test_oneshot)
    targets_test_oneshot = torch.cat(targets_test_oneshot)
    outputs_test_oneshot = outputs_test_oneshot.permute(1, 0)
    targets_test_oneshot = targets_test_oneshot.permute(1, 0)

    plt.figure(1)
    plt.title("Prediction image")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(outputs_test_oneshot.cpu())
    plt.savefig(folder + f"img_out_{target}.png")
    plt.show()

    plt.figure(2)
    plt.title("Target image")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(targets_test_oneshot.cpu())
    plt.savefig(folder + f"img_target_{target}.png")
    plt.show()

    # print the diff
    diff1 = abs(outputs_test_oneshot - targets_test_oneshot)
    diff1 = diff1 / diff1.max()

    plt.figure(3)
    plt.title("Differences between images - Target")
    plt.ylabel("section")
    plt.xlabel("timesteps(5min)")
    plt.imshow(torch.ones(diff1.shape) - diff1.cpu(), cmap="gray")
    plt.savefig(folder + f"diff_{target}.png")
    plt.show()

    plt.figure(4)
    plt.plot(losses_test_oneshot)
    plt.title("Training MSE")
    plt.ylabel("MSE")
    plt.xlabel("Bacthx100")
    plt.grid()
    plt.savefig(folder + f"train_mse_oneshot_{target}.png")
    plt.show()

    plt.figure(5)
    plt.plot(losses_test_oneshot2)
    # plt.ylim(0, 3)
    plt.title("Testing MAE")
    plt.ylabel("MAE")
    plt.xlabel("Bacth")
    plt.grid()
    plt.savefig(folder + f"test_mae_oneshot_{target}.png")
    plt.show()
