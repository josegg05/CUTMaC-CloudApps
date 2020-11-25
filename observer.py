import numpy as np
import pandas as pd
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Observer(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=20, n_layers=2):
        super(Observer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.layers[0] = nn.Linear(self.n_inputs, hidden_size)
        self.layers[-1] = nn.Linear(hidden_size, self.n_outputs)

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.layers[i](x)
            x = F.dropout(x, p=0.0)
        x = self.layers[-1](x)
        return x


class Observer2(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=20, n_layers=2):
        super(Observer2, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.layers_norm = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(self.n_layers - 1)])
        self.layers[0] = nn.Linear(self.n_inputs, hidden_size)
        self.layers[-1] = nn.Linear(hidden_size, self.n_outputs)

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.layers[i](x)
            x = F.dropout(x, p=0.4)
            x = self.layers_norm[i](x)
        x = self.layers[-1](x)
        return x


class Observer3(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=20, n_layers=2):
        super(Observer3, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.layers[0] = nn.Linear(self.n_inputs, hidden_size)
        self.layers[-1] = nn.Linear(hidden_size, self.n_outputs)

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.layers[i](x)
            x = torch.tanh(F.dropout(x, p=0.0))
        x = self.layers[-1](x)
        return x


class BranchWrapper(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=20, n_layers=2):
        super(BranchWrapper, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.upper_branch = Observer(n_inputs, n_outputs, hidden_size, n_layers)
        self.lower_branch = Observer(n_inputs, n_outputs, hidden_size, n_layers)

    def forward(self, x_upper, x_lower):
        x_upper = self.upper_branch(x_upper)
        x_lower = self.lower_branch(x_lower)

        return x_upper, x_lower


class ObserverRNN1(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=20, n_layers=2):
        super(ObserverRNN1, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 1
        self.gru = nn.GRU(input_size=self.n_inputs, hidden_size=self.hidden_size, num_layers=self.n_layers,
                          bidirectional=False, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.n_outputs)

        for name, param in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_normal_(param.data)

    def init_hidden(self, batch_size, hidden_size, gru):
        return torch.randn(self.n_layers * self.n_directions, batch_size, hidden_size).float().to(
            gru.all_weights[0][0].device)

    def forward(self, x):
        h = self.init_hidden(x.shape[0], self.hidden_size, self.gru)
        x, h = self.gru(x, h)
        return self.linear(x)


class ObserverRNN2(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=20, n_layers=2):
        super(ObserverRNN2, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 1
        self.gru = nn.GRU(input_size=self.n_inputs, hidden_size=self.hidden_size, num_layers=self.n_layers,
                          bidirectional=False, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.n_outputs)
        self.linear_out = nn.Linear(in_features=self.n_outputs, out_features=self.n_outputs)

        for name, param in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_normal_(param.data)

    def init_hidden(self, batch_size, hidden_size, gru):
        return torch.randn(self.n_layers * self.n_directions, batch_size, hidden_size).float().to(
            gru.all_weights[0][0].device)

    def forward(self, x):
        h = self.init_hidden(x.shape[0], self.hidden_size, self.gru)
        x, h = self.gru(x, h)
        x = torch.tanh(self.linear(x))
        return self.linear_out(x)


class BranchWrapperRecurrent(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=20, n_layers=2):
        super(BranchWrapperRecurrent, self).__init__()
        self.upper_branch = ObserverRNN1(n_inputs, n_outputs, hidden_size, n_layers)
        self.lower_branch = ObserverRNN1(n_inputs, n_outputs, hidden_size, n_layers)

    def forward(self, x_upper, x_lower):
        y_upper = self.upper_branch(x_upper)
        y_lower = self.lower_branch(x_lower)

        return y_upper, y_lower


class EncoderConv(nn.Module):
    def __init__(self, n_inputs, seqlen, kernel_sizes = [9, 7, 5], multiplier=2, n_output_hs=30, n_output_nl=1):
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
        self.pointwise_conv2 = nn.Conv1d(in_channels=n_inputs * multiplier, out_channels= n_inputs * multiplier,
                                         kernel_size=1, stride=1, groups=1)

        self.depthwise_conv3 = nn.Conv1d(in_channels=n_inputs * multiplier, out_channels=n_inputs * multiplier * 2,
                                         kernel_size=kernel_sizes[2], stride=1, groups=n_inputs)
        self.pointwise_conv3 = nn.Conv1d(in_channels=n_inputs * multiplier * 2, out_channels=n_inputs * multiplier * 2,
                                         kernel_size=1, stride=1, groups=1)

        channels = [n_inputs, n_inputs * multiplier, n_inputs * multiplier * 2]
        self.layerNorm1 = nn.LayerNorm(channels[0])
        self.layerNorm2 = nn.LayerNorm(channels[1])
        self.layerNorm3 = nn.LayerNorm(channels[2])

        Lout1 = seqlen
        Lout2 = Lout1 - (kernel_sizes[1] - 1) + 2 * 1  # el 2 * 1 es porque es padding de 1 a cada lado
        Lout3 = Lout2 - (kernel_sizes[2] - 1)

        # Salida para retransformar el espacio en un estado latene.
        self.linear1 = nn.Linear(in_features=Lout3, out_features=n_output_hs)
        self.linear2 = nn.Linear(in_features=channels[2], out_features=n_output_nl)

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
        # Nuevamente se normaliza en cada canal y vuelve al original
        # print(h.shape) # torch.Size([64, 66, 196])

        h = self.depthwise_conv3(h)
        # print(h.shape) # torch.Size([64, 132, 192])
        h = F.dropout(F.leaky_relu(self.pointwise_conv3(h)), p=0.2, training=self.training)
        # print(h.shape) # torch.Size([64, 132, 192])
        h = self.layerNorm3(h.transpose(1, 2)).transpose(1, 2) # torch.Size([64, 132, 192])

        # h_out_pred.shape = (batch size, n_inputs * multiplier * 2, Lout3)

        # Luego quiero retransofrmar esto a un estado latente

        # print(h.shape) # torch.Size([64, 132, 192])
        h = self.linear1(h)
        # print(h.shape) # torch.Size([64, 132, 30])
        h = self.linear2(h.transpose(1, 2))
        # print(h.shape) # torch.Size([64, 30, 5])
        h = h.transpose(0, 1).transpose(0, 2)

        # print(h.shape)
        return h


class EncoderRec(nn.Module):
    def __init__(self, n_inputs, hidden_size, num_layers):
        super(EncoderRec, self).__init__()
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = False
        self.n_directions = 1 if not self.bidirectional else 2

        self.gru = nn.GRU(input_size=self.n_inputs, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          bidirectional= self.bidirectional, batch_first=True, dropout=0.2)

    def init_hidden(self, batch_size, hidden_size, gru):
        return torch.randn(self.num_layers * self.n_directions, batch_size, hidden_size).float().to(
            gru.all_weights[0][0].device)

    def forward(self, input):
        h = self.init_hidden(batch_size=input.shape[0], hidden_size=self.hidden_size, gru=self.gru)
        x, h = self.gru(input, h)

        # print(h.shape)
        # h_out_pre.shape = (# of layers, batch size, hidden size)

        # print(x[-1, -1, :])
        # Cuando hay mas de una layer, el x son los h de la ultima capa
        # print()
        # print(h[-1, -1, :])
        return h


class DecoderNonLin(nn.Module):
    def __init__(self, n_outputs, hidden_size_in, num_layers_in):
        super(DecoderNonLin, self).__init__()
        self.linear1 = nn.Linear(in_features=num_layers_in, out_features=1)
        self.linear2 = nn.Linear(in_features=hidden_size_in, out_features=n_outputs)

    def forward(self, h):
        '''
        :param h: entra el espacio latente con forma (num_layers_rec, batch_size, hidden_size)
        :return y: voltaje cap salida con forma (batch_size, 1, n_output)
        '''

        h = h.transpose(0, 1)
        h = self.linear1(torch.tanh(h.transpose(1, 2)))
        # h.shape = (batch_size, hidden_size, 1)
        y = self.linear2(h.transpose(1, 2))

        return y


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
        :param h: shape = (num_layers_rec_enc, batch_size, hidden_size_rec_enc)
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


class EncoderDecoder1(nn.Module):
    '''
    Encoder conv
    Decoder non Lin
    '''
    def __init__(self, n_inputs, n_outputs, seqlen_conv, hidden_size=30, num_layers=1):
        super(EncoderDecoder1, self).__init__()

        self.encoder_c = EncoderConv(n_inputs=n_inputs, seqlen=seqlen_conv, n_output_hs=hidden_size,
                                     n_output_nl=num_layers)
        self.decoder_nl = DecoderNonLin(n_outputs=n_outputs, hidden_size_in=hidden_size, num_layers_in=num_layers)

    def forward(self, x_c, x_r=0):
        h = self.encoder_c(x_c)
        y = self.decoder_nl(h)
        return y


class EncoderDecoder2(nn.Module):
    '''
    Encoder conv
    Decoder rec
    '''
    def __init__(self, n_inputs, n_outputs, seqlen_conv, hidden_size, num_layers):
        super(EncoderDecoder2, self).__init__()

        self.encoder_c = EncoderConv(n_inputs=n_inputs, seqlen=seqlen_conv, n_output_hs=hidden_size,
                                     n_output_nl=num_layers)
        self.decoder_r = DecoderRec(n_inputs=n_inputs, n_outputs=n_outputs, hidden_size_in=hidden_size,
                                    num_layers=num_layers)

    def forward(self, x_c, x_r):
        h = self.encoder_c(x_c).contiguous()
        # print(x_c.shape, x_r.shape)
        y = self.decoder_r(x_r, h)
        return y


class EncoderDecoder3(nn.Module):
    '''
    Encoder sum( encoder conv, encoder rec)
    Decoder non Lin
    '''
    def __init__(self, n_inputs, n_outputs, seqlen_conv, hidden_size, num_layers):
        super(EncoderDecoder3, self).__init__()

        self.encoder_c = EncoderConv(n_inputs=n_inputs, seqlen=seqlen_conv, n_output_hs=hidden_size,
                                     n_output_nl=num_layers)
        self.encoder_r = EncoderRec(n_inputs=n_inputs, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder_nl = DecoderNonLin(n_outputs=n_outputs, hidden_size_in=hidden_size, num_layers_in=num_layers)

    def forward(self, x_c, x_r):
        h_c = self.encoder_c(x_c)
        h_r = self.encoder_r(x_r)
        y = self.decoder_nl(h_c + h_r)
        return y


class EncoderDecoder4(nn.Module):
    '''
    Encoder sum( encoder conv, encoder rec)
    Decoder rec
    '''
    def __init__(self, n_inputs, n_outputs, seqlen_conv, hidden_size, num_layers):
        super(EncoderDecoder4, self).__init__()

        self.encoder_c = EncoderConv(n_inputs=n_inputs, seqlen=seqlen_conv, n_output_hs=hidden_size,
                                     n_output_nl=num_layers)
        self.encoder_r = EncoderRec(n_inputs=n_inputs, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder_r = DecoderRec(n_inputs=n_inputs, n_outputs=n_outputs, hidden_size_in=hidden_size,
                                    num_layers=num_layers)


    def forward(self, x_c, x_r):
        h_c = self.encoder_c(x_c)
        h_r = self.encoder_r(x_r)
        y = self.decoder_r(x_r, h_c + h_r)
        return y


class EncoderDecoder5(nn.Module):
    '''
    Encoder concatenate( encoder conv, encoder rec)
    Decoder non Lin
    '''
    def __init__(self, n_inputs, n_outputs, seqlen_conv, hidden_size, num_layers):
        super(EncoderDecoder5, self).__init__()

        self.encoder_c = EncoderConv(n_inputs=n_inputs, seqlen=seqlen_conv, n_output_hs=hidden_size,
                                     n_output_nl=num_layers)
        self.encoder_r = EncoderRec(n_inputs=n_inputs, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.decoder_nl = DecoderNonLin(n_outputs=n_outputs, hidden_size_in=hidden_size, num_layers_in=num_layers)

    def forward(self, x_c, x_r):
        h_c = self.encoder_c(x_c)
        h_r = self.encoder_r(x_r)

        h = torch.cat((h_c, h_r), dim=2)
        h = self.linear(h)

        y = self.decoder_nl(h)
        return y


class EncoderDecoder6(nn.Module):
    '''
    Encoder concatenate( encoder conv, encoder rec)
    Decoder rec
    '''

    def __init__(self, n_inputs, n_outputs, seqlen_conv, hidden_size, num_layers):
        super(EncoderDecoder6, self).__init__()

        self.encoder_c = EncoderConv(n_inputs=n_inputs, seqlen=seqlen_conv, n_output_hs=hidden_size,
                                     n_output_nl=num_layers)
        self.encoder_r = EncoderRec(n_inputs=n_inputs, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.decoder_r = DecoderRec(n_inputs=n_inputs, n_outputs=n_outputs, hidden_size_in=hidden_size,
                                    num_layers=num_layers)

    def forward(self, x_c, x_r):
        h_c = self.encoder_c(x_c)
        h_r = self.encoder_r(x_r)

        h = torch.cat((h_c, h_r), dim=2)
        h = self.linear(h)

        y = self.decoder_r(x_r, h)
        return y


class EncoderDecoder7(nn.Module):
    '''
    Encoder concatenate( encoder conv, encoder rec) activation
    Decoder rec
    '''

    def __init__(self, n_inputs, n_outputs, seqlen_conv, hidden_size, num_layers):
        super(EncoderDecoder7, self).__init__()

        self.encoder_c = EncoderConv(n_inputs=n_inputs, seqlen=seqlen_conv, n_output_hs=hidden_size,
                                     n_output_nl=num_layers)
        self.encoder_r = EncoderRec(n_inputs=n_inputs, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.decoder_r = DecoderRec(n_inputs=n_inputs, n_outputs=n_outputs, hidden_size_in=hidden_size,
                                    num_layers=num_layers)

    def forward(self, x_c, x_r):
        h_c = self.encoder_c(x_c)
        h_r = self.encoder_r(x_r)

        h = torch.cat((h_c, h_r), dim=2)
        h = torch.tanh(self.linear(h))

        y = self.decoder_r(x_r, h)
        return y


if __name__ == "__main__":

    ##### Encoders testing

    n_inputs = 33
    seqlen_conv = 200
    seqlen_rec = 30
    hidden_size_rec = 30
    num_layers_rec = 2
    batch_size = 8

    enc_r = EncoderRec(n_inputs=n_inputs, hidden_size=hidden_size_rec, num_layers=num_layers_rec)
    enc_c = EncoderConv(n_inputs=n_inputs, seqlen=seqlen_conv, n_output_hs=hidden_size_rec, n_output_nl=num_layers_rec)
    X = torch.randn(batch_size, seqlen_conv, n_inputs)
    X_c = X
    X_r = X[:, -seqlen_rec:, :]

    h_c = enc_c(X_c)
    h_r = enc_r(X_r)

    print(h_c.shape, h_r.shape)

    # enc_r = Encoder_Rec(n_inputs=33, hidden_size=30, num_layers=2)
    # X = torch.randn(64, 20, 33)
    # enc_r(X)

    # enc = Encoder_Conv(n_inputs=33, seqlen=200)
    # X = torch.randn(64, 200, 33)
    # enc(X)

    ###### Decoder Non Lin testing
    dec_nl = DecoderNonLin(n_outputs=4, hidden_size_in=hidden_size_rec, num_layers_in=num_layers_rec)
    y = dec_nl(h_r)
    print(y.shape)

    ##### Decoder Recurrent testing
    dec_r = DecoderRec(n_inputs=n_inputs, n_outputs=4, hidden_size_in=hidden_size_rec, num_layers=num_layers_rec)
    y = dec_r(X_r, h_r)
    print(y.shape)


    ### Encoder Decoder RNN1
    # edRNN1 = ObserverRNN1(n_inputs=n_inputs, n_outputs=4, hidden_size=hidden_size_rec, n_layers=num_layers_rec)
    # y = edRNN1(X_r)
    # print(y.shape)
    '''
    #### Testing Encoder Decoder
    ed1 = EncoderDecoder1(n_inputs=n_inputs, n_outputs=4, seqlen_conv=seqlen_conv, hidden_size=hidden_size_rec,
                         num_layers=num_layers_rec)
    y = ed1(X_c)
    print(y.shape)
    '''
    ed2 = EncoderDecoder2(n_inputs=n_inputs, n_outputs=4, seqlen_conv=seqlen_conv, hidden_size=hidden_size_rec,
                          num_layers=num_layers_rec)


    y = ed2(X_c, X_r)
    print(y.shape)
    '''
    ed3 = EncoderDecoder3(n_inputs=n_inputs, n_outputs=4, seqlen_conv=seqlen_conv, hidden_size=hidden_size_rec,
                          num_layers=num_layers_rec)
    y = ed3(X_c, X_r)
    print(y.shape)

    ed4 = EncoderDecoder4(n_inputs=n_inputs, n_outputs=4, seqlen_conv=seqlen_conv, hidden_size=hidden_size_rec,
                          num_layers=num_layers_rec)

    y = ed4(X_c, X_r)
    print(y.shape)

    ed5 = EncoderDecoder5(n_inputs=n_inputs, n_outputs=4, seqlen_conv=seqlen_conv, hidden_size=hidden_size_rec,
                          num_layers=num_layers_rec)

    y = ed5(X_c, X_r)
    print(y.shape)

    ed6 = EncoderDecoder6(n_inputs=n_inputs, n_outputs=4, seqlen_conv=seqlen_conv, hidden_size=hidden_size_rec,
                          num_layers=num_layers_rec)

    y = ed6(X_c, X_r)
    print(y.shape)


    '''







