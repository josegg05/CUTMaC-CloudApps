import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

''' 
* Basic Network 
Model of a basic Network Composed by 2 Linear Layers
'''
class NN(nn.Module):
    def __init__(self, in_sensors=3, height=27, width=72, out_size=1):
        super(NN, self).__init__()

        # Create blocks
        self.linear1 = nn.Linear(in_sensors * height * width, 2048)
        self.drop = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear_out = nn.Linear(1024, out_size)

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = x.view(x.size(0), -1)
        #print(out.shape)
        out = nn.ReLU()(self.linear1(out))
        out = self.drop(out)
        out = nn.ReLU()(self.linear2(out))
        out = self.linear_out(out)
        return out


''' 
* CNN Network 
Model of a CNN Composed by 9 Residual Blocks (each with 2 convolutional
layers and batch normalization beween the layers), and 2 Linear Layers.
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.001, eps=0.001)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.001, eps=0.001)

        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same
        # as it's input, have a convolutional layer downsample the layer
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            diff = out_channels - in_channels
            if diff % 2 == 0:
                pad = (0, 0, 0, 0, int(diff / 2), int(diff / 2), 0, 0)
            else:
                pad = (0, 0, 0, 0, int(diff / 2), int(diff / 2) + 1, 0, 0)
            self.shortcut = nn.ZeroPad2d(pad)
        # if stride != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(
        #             in_channels=in_channels, out_channels=out_channels,
        #             kernel_size=(1, 1), stride=stride, bias=False
        #         ),
        #         nn.BatchNorm2d(out_channels)
        #     )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class CNN(nn.Module):
    def __init__(self, in_channels=3, height=27, width=72, out_size=1):
        super(CNN, self).__init__()

        # Create blocks
        self.block1 = self._create_block(in_channels, 32, stride=1)
        self.block2 = self._create_block(32, 64, stride=1)
        self.block3 = self._create_block(64, 96, stride=1)
        self.linear1 = nn.Linear(96 * height * width, 2048)
        self.drop = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear_out = nn.Linear(1024, out_size)

    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(out.size(0), -1)
        out = nn.ReLU()(self.linear1(out))
        out = self.drop(out)
        out = nn.ReLU()(self.linear2(out))
        out = self.linear_out(out)
        return out

''' 
* eRCNN 
Models of a Error Recurrent Convolutional Network Composed by a Convolutional Block
followed by a linear layer. The output of this block is then concatenated with the output
of a layer that receive the error of n previous predictions as input (error feedback), and then 
passed through a linear layers to finally produce a prediction 
'''
# Fist Version: Error initialization and recurrence have to be made out of the model code. It is not so
# efficient because the error have to be detached to allow the backpropagation. You can add some extra linear
# layers, but it doesn't affect to much the results
class eRCNN(nn.Module):
    def __init__(self, input_size, hid_error_size, output_size, pred_window=4, dev='cpu'):
        super().__init__()

        self.hid_error_size = hid_error_size
        last_in = 256+32
        self.pred_window = pred_window
        self.dev = dev
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1
        )
        self.lin_input = nn.Linear(12 * 35 * 32, 256)  # 32 (25*70) Feature maps after AvgPool2d(2)
        self.lin_error = nn.Linear(hid_error_size, 32)
        self.lin_out = nn.Linear(last_in, output_size)

    def forward(self, input, target):
        error = self.initError(input.shape[1])
        error = error.to(self.dev)
        pred_list = []
        for seq in range(input.shape[0]):
            out_in = nn.ReLU()(self.conv(input[seq]))
            out_in = nn.AvgPool2d(2)(
                out_in)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
            out_in = out_in.view(-1, self.num_flat_features(out_in))
            out_in = nn.ReLU()(self.lin_input(out_in))
            out_err = nn.ReLU()(self.lin_error(error))
            output = torch.cat((out_in, out_err), 1)
            output = self.lin_out(output)

            # Error vector Implementation 0 (BAD!!!)
            # err_seq = output - target[seq]
            # error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)

            # Error vector Implementation 1
            pred_list.append(output)
            if seq >= self.pred_window - 1:  # Hay que restarle uno porque lo estamos calculando un timestep antes de utilizarlo
                err_seq = pred_list[0] - target[seq - (self.pred_window - 1)]
                error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)
                pred_list.pop(0)

            # Error vector Implementation 2
            # if self.training:
            #     err_seq = output - target[seq]
            #     error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)
            # else:
            #     pred_list.append(output)
            #     if seq >= self.pred_window - 1:  # Implementation 2
            #         err_seq = pred_list[0] - target[seq - (self.pred_window - 1)]
            #         error = torch.cat((error[:, err_seq.shape[-1]:err_seq.shape[-1] * 3],
            #                            err_seq,
            #                            error[:, err_seq.shape[-1] * 3:]),
            #                           1)
            #         pred_list.pop(0)

        return output

    def initError(self, batch_size):
        return torch.zeros(batch_size, self.hid_error_size)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class eRCNN_fc_test(nn.Module):
    def __init__(self, input_size, hid_error_size, output_size, n_fc=0, fc_outs=[256]):
        super().__init__()

        self.hid_error_size = hid_error_size
        self.n_fc = n_fc
        last_in = 256+32
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1
        )
        self.lin_input = nn.Linear(12 * 35 * 32, 256)  # 32 (25*70) Feature maps after AvgPool2d(2)
        self.lin_error = nn.Linear(hid_error_size, 32)

        if self.n_fc >= 1:
            self.lin1 = nn.Linear(last_in, fc_outs[0])
            last_in = fc_outs[0]
        if self.n_fc >= 2:
            self.lin2 = nn.Linear(last_in, fc_outs[1])
            last_in = fc_outs[1]
        if self.n_fc >= 3:
            self.lin3 = nn.Linear(last_in, fc_outs[2])
            last_in = fc_outs[2]
        self.lin_out = nn.Linear(last_in, output_size)

    def forward(self, input, error):
        out_in = nn.ReLU()(self.conv(input))
        out_in = nn.AvgPool2d(2)(out_in)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
        out_in = out_in.view(-1, self.num_flat_features(out_in))
        out_in = nn.ReLU()(self.lin_input(out_in))
        out_err = nn.ReLU()(self.lin_error(error))
        output = torch.cat((out_in, out_err), 1)
        if self.n_fc >= 1:
            output = nn.ReLU()(self.lin1(output))
        if self.n_fc >= 2:
            output = nn.ReLU()(self.lin2(output))
        if self.n_fc >= 3:
            output = nn.ReLU()(self.lin3(output))
        output = self.lin_out(output)

        return output

    def initError(self, batch_size):
        return torch.zeros(batch_size, self.hid_error_size)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Newest Version: Error initialization and recurrence inside the code. Model is more efficient.
class eRCNNSeq(nn.Module):
    def __init__(self, input_size, hid_error_size, output_size, pred_window=None, out_seq=1, dev="cpu"):
        super().__init__()

        self.hid_error_size = hid_error_size
        self.out_seq = out_seq
        if pred_window is None:
            self.pred_window = out_seq
        else:
            self.pred_window = pred_window
        self.dev = dev
        last_in = 256+32
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1
        )
        self.lin_input = nn.Linear(12 * 35 * 32, 256)  # 32 (25*70) Feature maps after AvgPool2d(2)
        self.lin_error = nn.Linear(hid_error_size, 32)
        self.lin_out = nn.Linear(last_in, output_size)

    def forward(self, input, target):
        error = self.initError(input.shape[1])
        error = error.to(self.dev)
        out_list = []
        pred_list = []
        for seq in range(input.shape[0]):
            out_in = nn.ReLU()(self.conv(input[seq]))
            out_in = nn.AvgPool2d(2)(out_in)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
            out_in = out_in.view(-1, self.num_flat_features(out_in))
            out_in = nn.ReLU()(self.lin_input(out_in))
            out_err = nn.ReLU()(self.lin_error(error))
            output = torch.cat((out_in, out_err), 1)
            output = self.lin_out(output)

            # Error vector Implementation 0 (BAD!!!)
            # err_seq = output - target[seq]
            # error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)

            # Error vector Implementation 1
            pred_list.append(output)
            if seq >= self.pred_window - 1:  # Hay que restarle uno porque lo estamos calculando un timestep antes de utilizarlo
                err_seq = pred_list[0] - target[seq - (self.pred_window - 1)]
                error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)
                pred_list.pop(0)

            # Error vector Implementation 2
            # if self.training:
            #     err_seq = output - target[seq]
            #     error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)
            # else:
            #     pred_list.append(output)
            #     if seq >= self.pred_window - 1:  # Implementation 2
            #         err_seq = pred_list[0] - target[seq - (self.pred_window - 1)]
            #         error = torch.cat((error[:, err_seq.shape[-1]:err_seq.shape[-1] * 3],
            #                            err_seq,
            #                            error[:, err_seq.shape[-1] * 3:]),
            #                           1)
            #         pred_list.pop(0)

            if seq >= input.shape[0] - self.out_seq:
                out_list.append(output)

        output_final = torch.cat(out_list, 1)
        output_final = output_final.view(output_final.shape[0], self.out_seq, -1)
        #output_final = output_final.transpose(1, 2)
        return output_final

    def initError(self, batch_size):
        return torch.zeros(batch_size, self.hid_error_size)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class eRCNNSeqLin(nn.Module):
    def __init__(self, input_size, hid_error_size, output_size, pred_window, fc_pre_outs=[], dev="cpu", ):
        super().__init__()

        self.hid_error_size = hid_error_size
        self.pred_window = pred_window
        self.dev = dev
        self.n_fc = len(fc_pre_outs)
        last_in = 256+32
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1
        )
        self.lin_input = nn.Linear(12 * 35 * 32, 256)  # 32 (25*70) Feature maps after AvgPool2d(2)
        self.lin_error = nn.Linear(hid_error_size, 32)

        if self.n_fc >= 1:
            self.lin1 = nn.Linear(last_in, fc_pre_outs[0])
            last_in = fc_pre_outs[0]
        if self.n_fc >= 2:
            self.lin2 = nn.Linear(last_in, fc_pre_outs[1])
            last_in = fc_pre_outs[1]
        if self.n_fc >= 3:
            self.lin3 = nn.Linear(last_in, fc_pre_outs[2])
            last_in = fc_pre_outs[2]
        self.lin_out = nn.Linear(last_in, output_size)

    def forward(self, input, target):
        error = self.initError(input.shape[1])
        error = error.to(self.dev)
        pred_list = []
        for seq in range(input.shape[0]):
            out_in = nn.ReLU()(self.conv(input[seq]))
            out_in = nn.AvgPool2d(2)(out_in)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
            out_in = out_in.view(-1, self.num_flat_features(out_in))
            out_in = nn.ReLU()(self.lin_input(out_in))
            out_err = nn.ReLU()(self.lin_error(error))
            output = torch.cat((out_in, out_err), 1)
            if self.n_fc >= 1:
                output = nn.ReLU()(self.lin1(output))
            if self.n_fc >= 2:
                output = nn.ReLU()(self.lin2(output))
            if self.n_fc >= 3:
                output = nn.ReLU()(self.lin3(output))
            output = self.lin_out(output)

            # Error vector Implementation 0 (BAD!!!)
            # err_seq = output - target[seq]
            # error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)

            # Error vector Implementation 1
            pred_list.append(output)
            if seq >= self.pred_window - 1:  # Hay que restarle uno porque lo estamos calculando un timestep antes de utilizarlo
                err_seq = pred_list[0] - target[seq - (self.pred_window - 1)]
                error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)
                pred_list.pop(0)

            # Error vector Implementation 2
            # if self.training:
            #     err_seq = output - target[seq]
            #     error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)
            # else:
            #     pred_list.append(output)
            #     if seq >= self.pred_window - 1:  # Implementation 2
            #         err_seq = pred_list[0] - target[seq - (self.pred_window - 1)]
            #         error = torch.cat((error[:, err_seq.shape[-1]:err_seq.shape[-1] * 3],
            #                            err_seq,
            #                            error[:, err_seq.shape[-1] * 3:]),
            #                           1)
            #         pred_list.pop(0)

        return output

    def initError(self, batch_size):
        return torch.zeros(batch_size, self.hid_error_size)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class eRCNNSeqIter(nn.Module):
    def __init__(self, input_size, hid_error_size, output_size, pred_window=None, out_seq=1, dev="cpu"):
        super().__init__()

        self.hid_error_size = hid_error_size
        self.out_seq = out_seq
        if pred_window is None:
            self.pred_window = out_seq
        else:
            self.pred_window = pred_window
        self.dev = dev
        last_in = 256+32
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1
        )
        self.lin_input = nn.Linear(12 * 35 * 32, 256)  # 32 (25*70) Feature maps after AvgPool2d(2)
        self.lin_error = nn.Linear(hid_error_size, 32)
        self.lin_h = nn.Linear(last_in, 256)
        self.lin_out = nn.Linear(last_in, output_size)

    def forward(self, input, target, weight=100):
        error = self.initError(input.shape[1])
        error = error.to(self.dev)

        for seq in range(input.shape[0]):
            # Error vector Implementation 0 (Good - predictions is the next timestep)
            if seq != 0:  # not the first
                err_seq = output - target[seq - 1]
                error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)

            out_in = nn.ReLU()(self.conv(input[seq]))
            out_in = nn.AvgPool2d(2)(out_in)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
            out_in = out_in.view(-1, self.num_flat_features(out_in))
            out_in = nn.ReLU()(self.lin_input(out_in))
            out_err = nn.ReLU()(self.lin_error(error))
            output_pre = torch.cat((out_in, out_err), 1)
            output = self.lin_out(output_pre)

        out_list = []
        out_list.append(output)
        for seq in range(self.out_seq-1):
            if (not self.training) or (random.choices([True, False], weights=[weight, 100 - weight], k=1)[0]):
                # err_seq = torch.zeros(output.shape)
                # err_seq = err_seq.to(self.dev)
                err_seq = torch.mean(error.view(error.shape[0], 6, -1), 1)  # the "6" must come from outside
                print(err_seq.shape)
                print(error.shape)
            else:
                err_seq = output - target[seq - 1 + input.shape[0]]
            error = torch.cat((error[:, err_seq.shape[-1]:], err_seq), 1)

            out_h = self.lin_h(output_pre)
            out_err = nn.ReLU()(self.lin_error(error))
            output_pre = torch.cat((out_h, out_err), 1)
            output = self.lin_out(output_pre)
            out_list.append(output)

        output_final = torch.cat(out_list, 1)
        output_final = output_final.view(output_final.shape[0], self.out_seq, -1)
        #output_final = output_final.transpose(1, 2)
        return output_final

    def initError(self, batch_size):
        return torch.zeros(batch_size, self.hid_error_size)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


'''
* Encoder (Convolutional) Decoder (Recurrent)
Models of an Encoder Decoder, where the encoder is a Convolutional block (6 Convolutional
 layers) that receives a time sequence of all input signals to produce the initial hidden 
 states of the recurrent block. This block is a GRU based network that also receives a 
 time sequence of the input signals and the initial state of its hidden states from the 
 Encoder to generate a prediction or estimation
'''
class EncoderConv(nn.Module):
    def __init__(self, n_inputs, seqlen, kernel_sizes=[9, 7, 5], multiplier=2, n_output_hs=30,
                 n_output_nl=1):  # kernel_sizes=[9, 7, 5]
        super(EncoderConv, self).__init__()

        '''
        Conv1d : input (N, C_in, L_in) -> (N, C_out, L_out)
            N is batch size, C is channels and L is length of signal sequence

        n_output lo utilizamos para transformarlo en el espacio latente
        n_output_hs es el espacio latente del hidden size de encoder recurrent
        n_output_nl es la cantidad de layers que tiene un encoder recurrente

        '''

        padding1 = int((kernel_sizes[0] - 1) / 2)
        # padding avoid shrinking the image
        self.depthwise_conv1 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs, kernel_size=kernel_sizes[0],
                                         stride=1, groups=n_inputs, padding=padding1)
        self.pointwise_conv1 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs, kernel_size=1, stride=1, groups=1)

        self.depthwise_conv2 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs * multiplier,
                                         kernel_size=kernel_sizes[1], stride=1, groups=n_inputs, padding=1)
        self.pointwise_conv2 = nn.Conv1d(in_channels=n_inputs * multiplier, out_channels=n_inputs * multiplier,
                                         kernel_size=1, stride=1, groups=1)

        # self.depthwise_conv3 = nn.Conv1d(in_channels=n_inputs * multiplier, out_channels=n_inputs * multiplier * 2,
        #                                  kernel_size=kernel_sizes[2], stride=1, groups=n_inputs)
        # self.pointwise_conv3 = nn.Conv1d(in_channels=n_inputs * multiplier * 2, out_channels=n_inputs * multiplier * 2,
        #                                  kernel_size=1, stride=1, groups=1)

        channels = [n_inputs, n_inputs * multiplier, n_inputs * multiplier * 2]
        self.layerNorm1 = nn.LayerNorm(channels[0])
        self.layerNorm2 = nn.LayerNorm(channels[1])
        # self.layerNorm3 = nn.LayerNorm(channels[2])

        Lout1 = seqlen
        Lout2 = Lout1 - (kernel_sizes[1] - 1) + 2 * 1  # el 2 * 1 es porque es padding de 1 a cada lado
        Lout3 = Lout2 - (kernel_sizes[2] - 1)

        # Salida para retransformar el espacio en un estado latene.
        self.linear1 = nn.Linear(in_features=Lout2, out_features=n_output_hs)  # in_features=Lout3
        self.linear2 = nn.Linear(in_features=channels[1], out_features=n_output_nl)  # in_features=channels[2]

    def forward(self, input):
        x = input
        h = self.depthwise_conv1(x.transpose(1, 2))
        # print(h.shape) # torch.Size([64, 33, 200])
        h = F.dropout(F.leaky_relu(self.pointwise_conv1(h)), p=0.2, training=self.training)
        # print(h.shape) # torch.Size([64, 33, 200])
        # Hasta acá realiza operaciones matematicas pero mantiene el tamaño de
        # canales y de largo de secuencia
        h = self.layerNorm1(h.transpose(1, 2)).transpose(1, 2)
        # Normaliza en torno a los canales y luego vuelve a la posicion original
        # print(h.shape) # torch.Size([64, 33, 200])

        h = self.depthwise_conv2(h)
        # print(h.shape) # torch.Size([64, 66, 196])
        h = F.dropout(F.leaky_relu(self.pointwise_conv2(h)), p=0.2, training=self.training)
        # print(h.shape) # torch.Size([64, 66, 196])
        h = self.layerNorm2(h.transpose(1, 2)).transpose(1, 2)
        #  Nuevamente se normaliza en cada canal y vuelve al original
        # print(h.shape) # torch.Size([64, 66, 196])

        # h = self.depthwise_conv3(h)
        # # print(h.shape) # torch.Size([64, 132, 192])
        # h = F.dropout(F.leaky_relu(self.pointwise_conv3(h)), p=0.2, training=self.training)
        # # print(h.shape) # torch.Size([64, 132, 192])
        # h = self.layerNorm3(h.transpose(1, 2)).transpose(1, 2) # torch.Size([64, 132, 192])

        # h_out_pred.shape = (batch size, n_inputs * multiplier * 2, Lout3)

        # Luego quiero retransofrmar esto a un estado latente

        # print(h.shape) # torch.Size([64, 132, 192])
        h = self.linear1(h)
        # print(h.shape) # torch.Size([64, 132, 20])
        h = self.linear2(h.transpose(1, 2))
        # print(h.shape) # torch.Size([64, 20, 2])
        h = h.transpose(0, 1).transpose(0, 2)
        # print(h.shape) # torch.Size([2, 64, 20]) = [num_layers_rec_dec, batch_size, hidden_size_rec]
        # in my case = torch.Size([2, 50, 7])

        # print(h.shape)
        return h


class EncoderConv2D(nn.Module):
    def __init__(self, n_inputs, n_output_hs=30,  n_output_nl=1):
        super(EncoderConv2D, self).__init__()

        '''
        Conv1d : input (N, C_in, L_in) -> (N, C_out, L_out)
            N is batch size, C is channels and L is length of signal sequence

        n_output lo utilizamos para transformarlo en el espacio latente
        n_output_hs es el espacio latente del hidden size de encoder recurrent
        n_output_nl es la cantidad de layers que tiene un encoder recurrente

        '''
        conv_out_channels = 32
        self.conv = nn.Conv2d(
            in_channels=n_inputs,
            out_channels=conv_out_channels,
            kernel_size=(3, 3),
            stride=1
        )

        # Salida para retransformar el espacio en un estado latene.
        self.linear1 = nn.Linear(in_features=12 * 35, out_features=n_output_hs)  # in_features=Lout3
        self.linear2 = nn.Linear(in_features=conv_out_channels, out_features=n_output_nl)  # in_features=channels[2]

    def forward(self, input):
        h = nn.ReLU()(self.conv(input))
        h = nn.AvgPool2d(2)(h)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
        h = h.view(h.shape[0], h.shape[1], -1)  # h = h.view(-1, h.shape[1], self.num_flat_features(h))

        # Luego quiero retransofrmar esto a un estado latente
        # print(h.shape) # torch.Size([50, 32, n])
        h = self.linear1(h)
        # print(h.shape) # torch.Size([50, 32, 7])
        h = self.linear2(h.transpose(1, 2))
        # print(h.shape) # torch.Size([50, 7, 2])
        h = h.transpose(0, 1).transpose(0, 2)
        # print(h.shape) # torch.Size([2, 50, 7]) = [num_layers_rec_dec, batch_size, hidden_size_rec]

        # print(h.shape)
        return h

    def num_flat_features(self, x):
        size = x.size()[2:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EncoderConv2DL(nn.Module):
    def __init__(self, n_inputs, n_output_hs=30,  n_output_nl=1, first_lin_size=256):
        super(EncoderConv2DL, self).__init__()

        '''
        Conv1d : input (N, C_in, L_in) -> (N, C_out, L_out)
            N is batch size, C is channels and L is length of signal sequence

        n_output lo utilizamos para transformarlo en el espacio latente
        n_output_hs es el espacio latente del hidden size de encoder recurrent
        n_output_nl es la cantidad de layers que tiene un encoder recurrente

        '''
        conv_out_channels = 32
        self.conv = nn.Conv2d(
            in_channels=n_inputs,
            out_channels=conv_out_channels,
            kernel_size=(3, 3),
            stride=1
        )

        # Salida para retransformar el espacio en un estado latene.
        self.linear1 = nn.Linear(in_features=12 * 35, out_features=first_lin_size)  # in_features=Lout3
        self.linear2 = nn.Linear(in_features=first_lin_size, out_features=n_output_hs)
        self.linear3 = nn.Linear(in_features=conv_out_channels, out_features=n_output_nl)  # in_features=channels[2]

    def forward(self, input):
        h = nn.ReLU()(self.conv(input))
        h = nn.AvgPool2d(2)(h)  # Average Pooling with a square kernel_size=(2,2) and stride=kernel_size=(2,2)
        h = h.view(h.shape[0], h.shape[1], -1)  # h = h.view(-1, h.shape[1], self.num_flat_features(h))

        # Luego quiero retransofrmar esto a un estado latente
        # print(h.shape) # torch.Size([50, 32, n])
        h = self.linear1(h)
        # print(h.shape) # torch.Size([50, 32, first_lin_size])
        h = self.linear2(h)
        # print(h.shape) # torch.Size([50, 32, 7])
        h = self.linear3(h.transpose(1, 2))
        # print(h.shape) # torch.Size([50, 7, 2])
        h = h.transpose(0, 1).transpose(0, 2)
        # print(h.shape) # torch.Size([2, 50, 7]) = [num_layers_rec_dec, batch_size, hidden_size_rec]

        # print(h.shape)
        return h

    def num_flat_features(self, x):
        size = x.size()[2:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class DecoderRec(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size_in, num_layers):
        super(DecoderRec, self).__init__()

        self.gru = nn.GRU(input_size=n_inputs, hidden_size=hidden_size_in, num_layers=num_layers, batch_first=True,
                          bidirectional=False, dropout=0.2)
        self.linear1 = nn.Linear(in_features=num_layers, out_features=1)
        self.linear2 = nn.Linear(in_features=hidden_size_in, out_features=n_outputs)

    def forward(self, x, h):
        '''

        :param x: shape = (batch_size, seqlen, n_inputs)
        :param h: shape = (num_layers_rec_dec, batch_size, hidden_size_rec_dec)
        :return: y: shape = (batch_size, 1, n_outputs)
        '''

        x, h = self.gru(x, h)
        h = h.transpose(0, 1)
        # h = self.linear1(h.transpose(1, 2))
        h = h.transpose(1, 2)
        size = h.shape
        h = h[:, :, -1].view(size[0], size[1], 1)
        # h.shape = (batch_size, hidden_size, 1)
        y = self.linear2(h.transpose(1, 2))
        return y


class DecoderRecL(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size_in, num_layers):
        super(DecoderRecL, self).__init__()

        self.gru = nn.GRU(input_size=n_inputs, hidden_size=hidden_size_in, num_layers=num_layers, batch_first=True,
                          bidirectional=False, dropout=0.2)
        self.linear1 = nn.Linear(in_features=num_layers, out_features=1)
        self.linear2 = nn.Linear(in_features=hidden_size_in, out_features=n_outputs)

    def forward(self, x, h):
        '''

        :param x: shape = (batch_size, seqlen, n_inputs)
        :param h: shape = (num_layers_rec_dec, batch_size, hidden_size_rec_dec)
        :return: y: shape = (batch_size, 1, n_outputs)
        '''

        x, h = self.gru(x, h)
        h = h.transpose(0, 1)
        h = self.linear1(h.transpose(1, 2))
        #h = h.transpose(1, 2)
        #size = h.shape
        #h = h[:, :, -1].view(size[0], size[1], 1)
        # h.shape = (batch_size, hidden_size, 1)
        y = self.linear2(h.transpose(1, 2))
        return y


class ErrorDecoderRec(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size_in, num_layers, out_seq=3, error_size=6, linear_emb=False, dev='cpu'):
        super(ErrorDecoderRec, self).__init__()

        self.out_seq = out_seq
        self.error_size = error_size
        self.detectors_pred = n_outputs
        self.linear_emb = linear_emb
        self.dev = dev
        self.gru = nn.GRU(input_size=n_inputs, hidden_size=hidden_size_in, num_layers=num_layers, batch_first=True,
                          bidirectional=False, dropout=0.2)
        self.linear1 = nn.Linear(in_features=num_layers, out_features=out_seq)
        self.linear2 = nn.Linear(in_features=hidden_size_in, out_features=n_outputs)

    def forward(self, e, h, t):
        '''

        :param x: shape = (batch_size, seqlen, n_inputs)
        :param h: shape = (num_layers_rec_dec, batch_size, hidden_size_rec_dec)
        :return: y: shape = (batch_size, 1, n_outputs)
        '''


        #aux = torch.zeros(e.shape[0], self.out_seq-1, self.detectors_pred) # Option 1
        e_seq = e
        for i in range(self.out_seq-1):
            aux = torch.mean(e_seq[:, i:, :], 1)
            aux = torch.unsqueeze(aux, 1)
            #aux = aux.to(self.dev)
            e_seq = torch.cat((e_seq, aux), 1)
        e_seq = e_seq.to(self.dev)
        y, h = self.gru(e_seq, h)  # y.shape --> (batch_size, seq_len, hidden_size)

        if(self.linear_emb):
            y = self.linear1(y.transpose(0, 1).transpose(1, 2))
            y = y.transpose(1, 2)
        else:
            size = y.shape
            y = y[:, -self.out_seq:, :].view(size[0], self.out_seq, size[2])
        # y.shape --> (batch_size, out_seq, hidden_size)
        y = self.linear2(y)

        #print(t.shape)
        #print(y[:, -1, :].shape)
        e_last = t - y[:, -1, :]
        #e = torch.cat((e[:, 1:self.error_size, :], torch.unsqueeze(e_last, 1), e[:, -(self.out_seq-1):, :]), 1)
        e = torch.cat((e[:, 1:, :], torch.unsqueeze(e_last, 1)), 1)

        return y, e


    def error_init(self, batch_size):
        #error = torch.zeros(batch_size, self.error_size + self.out_seq - 1, self.detectors_pred)
        error = torch.zeros(batch_size, self.error_size, self.detectors_pred)
        error = error.to(self.dev)
        return error


class EncoderDecoder(nn.Module):
    '''
    Encoder conv
    Decoder rec
    '''

    def __init__(self, n_inputs, n_outputs, seqlen_conv, hidden_size, num_layers):
        super(EncoderDecoder, self).__init__()

        self.encoder_c = EncoderConv(n_inputs=n_inputs, seqlen=seqlen_conv, n_output_hs=hidden_size,
                                     n_output_nl=num_layers)
        self.decoder_r = DecoderRec(n_inputs=n_inputs, n_outputs=n_outputs, hidden_size_in=hidden_size,
                                    num_layers=num_layers)

    def forward(self, x_c, x_r):
        h = self.encoder_c(x_c).contiguous()
        # print(x_c.shape, x_r.shape)
        y = self.decoder_r(x_r, h)
        return y


class EncoderDecoder2D(nn.Module):
    '''
    Encoder conv
    Decoder rec
    '''

    def __init__(self, n_inputs_enc, n_inputs_dec, n_outputs, hidden_size, num_layers, lin1_conv=False, lin1_rec=False,
                 first_lin_size=256):
        super(EncoderDecoder2D, self).__init__()
        if lin1_conv:
            self.encoder_c = EncoderConv2DL(n_inputs=n_inputs_enc, n_output_hs=hidden_size, n_output_nl=num_layers,
                                            first_lin_size=first_lin_size)
        else:
            self.encoder_c = EncoderConv2D(n_inputs=n_inputs_enc, n_output_hs=hidden_size, n_output_nl=num_layers)

        if lin1_rec:
            self.decoder_r = DecoderRecL(n_inputs=n_inputs_dec, n_outputs=n_outputs, hidden_size_in=hidden_size,
                                         num_layers=num_layers)
        else:
            self.decoder_r = DecoderRec(n_inputs=n_inputs_dec, n_outputs=n_outputs, hidden_size_in=hidden_size,
                                        num_layers=num_layers)

    def forward(self, x_c, x_r):
        h = self.encoder_c(x_c).contiguous()
        # print(x_c.shape, x_r.shape)
        y = self.decoder_r(x_r, h)
        return y


class ErrorEncoderDecoder2D(nn.Module):
    '''
    Encoder conv
    Decoder rec
    '''

    def __init__(self, n_inputs_enc, n_inputs_dec, n_outputs, hidden_size, num_layers,
                 out_seq=3, error_size=6, lin1_conv=False, lin1_rec=False, first_lin_size=256, dev='cpu'):
        super(ErrorEncoderDecoder2D, self).__init__()

        self.dev = dev
        if lin1_conv:
            self.encoder_c = EncoderConv2DL(n_inputs=n_inputs_enc, n_output_hs=hidden_size, n_output_nl=num_layers,
                                            first_lin_size=first_lin_size)
        else:
            self.encoder_c = EncoderConv2D(n_inputs=n_inputs_enc, n_output_hs=hidden_size, n_output_nl=num_layers)

        self.decoder_r = ErrorDecoderRec(n_inputs=n_inputs_dec, n_outputs=n_outputs, hidden_size_in=hidden_size,
                                         num_layers=num_layers, out_seq=out_seq, error_size=error_size,
                                         linear_emb=lin1_rec, dev=dev)

    def forward(self, x_c, target):
        error = self.decoder_r.error_init(x_c.shape[1])
        error = error.to(self.dev)

        for idx in range(len(x_c)):
            h = self.encoder_c(x_c[idx]).contiguous()
            # print(x_c.shape, x_r.shape)
            y, error = self.decoder_r(error, h, target[idx])
        return y
