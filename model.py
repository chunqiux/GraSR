import torch
from torch import nn


class Encoder(nn.Module):
    r"""This class encode the raw features to the descriptors

    Args:
        n_ref_points (int): the number of reference points

    Shape:
        - x_o: :math:`[B, N, n_ref + 1]`
        - ld: :math:`B`
        - am: :math:`[B, L_\text{max} \times L_\text{max}, N]`
        - Output: :math:`[\sum^B_i{L_i}, N, 1, 1]`
    where B is the batch size, N is the size of the node feature vector, n_ref is the number of reference points
    , :math:`L_i` is the length of the i-th sequence, and :math:`L_\text{max}`
    is the length of the longest sequence in the batch.
    """
    def __init__(self, n_ref_points):
        super().__init__()
        mlp1_dim = 64
        self.mlp1 = nn.Sequential(
            nn.Conv2d(1, mlp1_dim, (1, n_ref_points + 1)),
            nn.BatchNorm2d(mlp1_dim),
            nn.LeakyReLU(1e-2, inplace=True),
        )

        cell_size = 64
        self.bilstm = nn.LSTM(mlp1_dim, cell_size, batch_first=True, bidirectional=True)

        mlp2_dim = 256
        self.mlp2 = nn.Sequential(
            nn.Conv2d(cell_size * 2, mlp2_dim, 1),
            nn.BatchNorm2d(mlp2_dim),
            nn.LeakyReLU(1e-2, inplace=True),
        )

        gcrb_dim = [256, 512]
        self.gcl = GraphConvLayer(mlp2_dim, gcrb_dim[0], activation='leaky_relu')
        self.gcrb_1 = GraphConvResBlock(gcrb_dim[0], gcrb_dim[0])
        self.gcrb_2 = GraphConvResBlock(gcrb_dim[0], gcrb_dim[1])

        self.fc = nn.Sequential(
            nn.Conv2d(gcrb_dim[-1], 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(512, 400, 1),
        )

    def rnn_module(self, x, ld):
        x_1 = x.unsqueeze(1)  # [B, 1, N, 32]
        x_2 = self.mlp1(x_1)  # [B, 64, N, 1]
        x_2 = x_2.squeeze(3).permute(0, 2, 1)  # [B, N, 64]

        px = nn.utils.rnn.pack_padded_sequence(x_2, ld, enforce_sorted=False, batch_first=True)
        packed_out, (ht, ct) = self.bilstm(px)
        padded_out = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, padding_value=-float('inf'))

        group_1 = []
        for i, len_seq in enumerate(ld):
            y = padded_out[0][i, :len_seq, :]
            group_1.append(y)
        x_3 = torch.cat(group_1, dim=0)
        x_3 = x_3.unsqueeze(2).unsqueeze(3)
        x_4 = self.mlp2(x_3)  # [B, 256, 1, 1]
        return x_4

    def forward(self, x_o, ld, am):
        x = self.rnn_module(x_o, ld)

        x = self.gcl(x, ld, am)
        x = self.gcrb_1(x, ld, am)
        x = self.gcrb_2(x, ld, am)

        group_2 = []
        prev = 0
        for i, len_seq in enumerate(ld):
            y = x[prev:prev + len_seq, :]
            g_max, _ = torch.max(y, 0)
            group_2.append(g_max)
            prev += len_seq
        x_1 = torch.stack(group_2, 0)  # [B, 512, 1, 1]
        x_2 = self.fc(x_1)
        x_2 = x_2.squeeze(-1).squeeze(-1)

        return x_2


class GraphConvResBlock(nn.Module):
    r"""This class applies a residual block containing two GraphConvLayers

    Args:
        input_size (int): The size of input node feature vector
        output_size (int): The size of output node feature vector

    Shape:
        - x: :math:`[\sum^B_i{L_i}, N, 1, 1]`
        - ld: :math:`B`
        - am: :math:`[B, L_\text{max} \times L_\text{max}, N]`
        - Output: :math:`[\sum^B_i{L_i}, N, 1, 1]`
    where B is the batch size, N is the size of the node feature vector,
    , :math:`L_i` is the length of the i-th sequence, and :math:`L_\text{max}`
    is the length of the longest sequence in the batch.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Sequential(
            nn.Conv2d(self.input_size, self.output_size, 1),
            nn.BatchNorm2d(self.output_size)
        )

        self.graph_conv_layer_1 = GraphConvLayer(input_size, input_size, 0.2, 'relu')
        self.graph_conv_layer_2 = GraphConvLayer(input_size, output_size)
        self.activate = nn.LeakyReLU(1e-2, inplace=True)

    def forward(self, x, ld, am):
        # linear transformation for input
        residual = self.linear(x) if self.input_size != self.output_size else x
        # two GCN layers
        x = self.graph_conv_layer_1(x, ld, am)
        x = self.graph_conv_layer_2(x, ld, am)
        x = self.activate(x + residual)
        return x


class GraphConvLayer(nn.Module):
    r"""This class applies a graph convolutional layer

    Args:
        input_size (int): The size of input node feature vector
        output_size (int): The size of output node feature vector
        dropout (float): a dropout layer after MLP. Default=0.0
        activation (str, optional): an activation function (``'relu'`` or ``'leaky_relu'``) after MLP. Default=None.

    Shape:
        - x: :math:`[\sum^B_i{L_i}, N, 1, 1]`
        - ld: :math:`B`
        - am: :math:`[B, L_\text{max} \times L_\text{max}, N]`
        - Output: :math:`[\sum^B_i{L_i}, N, 1, 1]`
    where B is the batch size, N is the size of the node feature vector,
    , :math:`L_i` is the length of the i-th sequence, and :math:`L_\text{max}`
    is the length of the longest sequence in the batch.
    """

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0, activation: str = None):
        super(GraphConvLayer, self).__init__()

        self.nonlinear = nn.Sequential(
            nn.Conv2d(input_size, output_size, 1),
            nn.BatchNorm2d(output_size)
        )

        if activation == 'relu':
            self.nonlinear.add_module('Relu', nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            self.nonlinear.add_module('Leaky_Relu', nn.LeakyReLU(inplace=True))
        if dropout > 0:
            self.nonlinear.add_module('Dropout', nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, ld: list, am: torch.Tensor) -> torch.Tensor:
        x = aggregate_node(x, ld, am)
        x = self.nonlinear(x)
        return x


def aggregate_node(x: torch.Tensor, ld: list, am: torch.Tensor) -> torch.Tensor:
    r"""
    This function is used to aggregate node features in the graph.

    :param x:  node feature matrix
    :param ld:  sequence length list
    :param am:  adjacency matrix
    :return:  updated node feature matrix

    Shape:
        - x: :math:`[\sum^B_i{L_i}, N, 1, 1]`
        - ld: :math:`B`
        - am: :math:`[B, L_\text{max} \times L_\text{max}, N]`
        - return: :math:`[\sum^B_i{L_i}, N, 1, 1]`
    where B is the batch size, N is the size of the node feature vector,
    , :math:`L_i` is the length of the i-th sequence, and :math:`L_\text{max}`
    is the length of the longest sequence in the batch.
    """
    x = x.squeeze()
    prev, ba_group = 0, []
    for j, len_seq in enumerate(ld):
        y = x[prev:prev + len_seq, :]  # [l, input_size]
        single_am = am[j, :len_seq, :len_seq]
        y = single_am @ y  # [l, input_size]
        ba_group.append(y)
        prev += len_seq
    x = torch.cat(ba_group, 0)  # [B, input_size]
    x = x.unsqueeze(2).unsqueeze(3)  # [B, input_size, 1, 1]
    return x
