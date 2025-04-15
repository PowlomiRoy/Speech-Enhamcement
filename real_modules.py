import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from real_utils import *
from cplxmodule.nn import CplxConv2d, CplxConvTranspose2d, CplxBatchNorm2d, CplxLinear, CplxConv1d, CplxBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(9999)
EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)

# -----------------------   Architecture Parameters --------------------------------

# Step: 1.1 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Encoder/Decoder >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False, DSC=False):
        super(Encoder, self).__init__()
        # DSC: depthwise_separable_conv
        if DSC:
            self.conv = DSC_Encoder(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.relu(self.norm(self.conv(x)))


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), output_padding=(0, 0), bias=False, DSC=False):
        super(Decoder, self).__init__()
        # DSC: depthwise_separable_conv
        if DSC:
            self.conv = DSC_Decoder(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.relu(self.norm(self.conv(x)))
    
class DSC_Encoder(nn.Module):
    # depthwise_separable_conv
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=(1, 1), bias=False):
        super(DSC_Encoder, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)  # group = in_ch; and in_ch=out_ch
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)  # Kernel_size = 1 always

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DSC_Decoder(nn.Module):
    # depthwise_separable_conv
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=(1, 1), output_padding=(0, 0),
                 bias=False):
        super(DSC_Decoder, self).__init__()
        self.depthwise = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, groups=in_channels, output_padding=output_padding,
                                            bias=bias)  # group = in_ch; and in_ch=out_ch
        self.pointwise = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1,
                                            bias=bias)  # Kernel_size = 1 always

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
# Step: 1.2 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Frequency Transformation Block >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class NodeReshape(nn.Module):
    def __init__(self, shape):
        super(NodeReshape, self).__init__()
        self.shape = shape

    def forward(self, feature_in: torch.Tensor):
        shape = feature_in.size()
        batch = shape[0]
        new_shape = [batch]
        new_shape.extend(list(self.shape))
        return feature_in.reshape(new_shape)

class Freq_FC(nn.Module):
    def __init__(self, F_dim, bias=False):
        super(Freq_FC, self).__init__()
        self.linear = nn.Linear(F_dim, F_dim, bias=bias)

    def forward(self, x):
        out = x.transpose(-1, -2).contiguous()  # [B, C, T, F] -> [B, C, F, T]
        out = self.linear(out)
        out = out.transpose(-1, -2).contiguous()  # Back to [B, C, T, F]
        return out


class RealFTB(nn.Module):
    """Real-valued Frequency-Time Block (FTB)"""

    def __init__(self, F_dim, channels):
        super(RealFTB, self).__init__()
        self.channels = channels
        self.C_r = 5
        self.F_dim = F_dim  # This will get updated dynamically in forward

        self.Conv2D_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.C_r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.C_r),
        )

        self.Conv1D_1 = nn.Sequential(
            nn.Conv1d(self.F_dim * self.C_r, self.F_dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(self.F_dim),
        )

        self.FC = Freq_FC(F_dim, bias=False)

        self.Conv2D_2 = nn.Sequential(
            nn.Conv2d(2 * self.channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
        )

        self.att_inner_reshape = NodeReshape([self.F_dim * self.C_r, -1])
        self.att_out_reshape = NodeReshape([1, F_dim, -1])

    def cat(self, x, y, dim):
        return torch.cat([x, y], dim)

    def forward(self, inputs, verbose=False):
        # feature_n: [batch, channel_in_out, T, F]

        _, _, self.F_dim, self.T_dim = inputs.shape
        # Conv2D
        out = torch.relu(self.Conv2D_1(inputs));
        if verbose: print('Layer-1               : ', out.shape)  # [B,Cr,T,F]
        # Reshape: [batch, channel_attention, F, T] -> [batch, channel_attention*F, T]
        out = out.view(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])
        # out = self.att_inner_reshape(out);
        if verbose: print('Layer-2               : ', out.shape)
        # out = out.view(-1, self.T_dim, self.F_dim * self.C_r) ; print(out.shape) # [B,c_ftb_r*f,segment_length]
        # Conv1D
        out = torch.relu(self.Conv1D_1(out));
        if verbose: print('Layer-3               : ', out.shape)  # [B,F, T]
        # temp = self.att_inner_reshape(temp); print(temp.shape)
        out = out.unsqueeze(1)
        # out = out.view(-1, self.channels, self.F_dim, self.T_dim);
        if verbose: print('Layer-4               : ', out.shape)  # [B,c_a,segment_length,1]
        # Multiplication with input
        out = out * inputs;
        if verbose: print('Layer-5               : ', out.shape)  # [B,c_a,segment_length,1]*[B,c_a,segment_length,f]
        # Frequency- FC
        # out = torch.transpose(out, 2, 3)  # [batch, channel_in_out, T, F]
        out = self.FC(out);
        # if verbose: print('Layer-6               : ', out.shape)  # [B,c_a,segment_length,f]
        # out = torch.transpose(out, 2, 3)  # [batch, channel_in_out, T, F]
        # Concatenation with Input
        out = self.cat(out, inputs, 1);
        if verbose: print('Layer-7               : ', out.shape)  # [B,2*c_a,segment_length,f]
        # Conv2D
        outputs = torch.relu(self.Conv2D_2(out));
        if verbose: print('Layer-8               : ', outputs.shape)  # [B,c_a,segment_length,f]

        return outputs




# -------------------------------- Depth wise Seperable Convolution --------------------------------

class depthwise_separable_convx(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_convx, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
# Step: 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Skip Connection >>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, DSC=False):
        super(SkipBlock, self).__init__()
        # DSC: depthwise_separable_conv

        if DSC:
            self.conv = DSC_Encoder(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x))) + x


class SkipConnection(nn.Module):
    """
    SkipConnection is a concatenation of SkipBlocks
    """

    def __init__(self, in_channels, num_convblocks, DSC=False):
        super(SkipConnection, self).__init__()
        self.skip_blocks = nn.Sequential(*[
            SkipBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, DSC=DSC)
            for _ in range(num_convblocks)
        ])

    def forward(self, x):
        return self.skip_blocks(x)
    
# Step: 3 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Activation Function >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def real_mul(x, y):
    return x * y

def real_sigmoid(input):
    return torch.sigmoid(input)

class real_softplus(nn.Module):
    def __init__(self):
        super(real_softplus, self).__init__()
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, input):
        return self.softplus(input)

class real_elu(nn.Module):
    def __init__(self):
        super(real_elu, self).__init__()
        self.elu = nn.ELU(inplace=False)

    def forward(self, input):
        return self.elu(input)
    
# Step: 4 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Bottleneck layers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step: 4.1 --------------------  GRU layers --------------------------

class RealGRU(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(RealGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=input_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.select_output = SelectItem(0)  # Assuming SelectItem selects the GRU output
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.transpose(-1, -2).contiguous()
        out, _ = self.gru(x)
        out = self.select_output((out,))  # If SelectItem is written for tuple input
        out = self.linear(out)
        out = out.transpose(-1, -2)
        return out
    
# Step: 4.2 --------------------  Complex Transformer layers --------------------------

class Transformer_single(nn.Module):
    def __init__(self, nhead=8):
        super(Transformer_single, self).__init__()
        self.nhead = nhead

    def forward(self, x):
        # x = torch.randn(10, 2, 80, 256) [batch, Ch, F, T]
        b, c, F, T = x.shape
        STB = TransformerEncoderLayer(d_model=F, nhead=self.nhead)  # d_model = Expected feature
        STB.to("cpu")
        x = x.permute(1, 0, 3, 2).contiguous().view(-1, b * T, F)  # [c, b*T, F]
        x = x.to("cpu")
        x = STB(x)
        x = x.view(b, c, F, T)  # [b, c, F, T]
        return x


class Transformer_multi(nn.Module):
    # d_model = x.shape[3]
    def __init__(self, nhead, layer_num=2):
        super(Transformer_multi, self).__init__()
        self.layer_num = layer_num
        self.MTB = Transformer_single(nhead=nhead)  # d_model: the number of expected features in the input

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.MTB(x)
        return x
    
class RealTransformer(nn.Module):
    def __init__(self, nhead, num_layer):
        super(RealTransformer, self).__init__()
        self.trans = Transformer_multi(nhead=nhead, layer_num=num_layer)  # Uses only real part

    def forward(self, x):
        out = self.trans(x)
        return out

class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        device = torch.device("cpu")  # Force CPU usage
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout).to(device)
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)
        if bidirectional:
            self.linear2 = Linear(d_model * 2 * 2, d_model)
        else:
            self.linear2 = Linear(d_model * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # print("Tensor src evice:", src.device)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# Step: 4.3 --------------------  Complex DPRNN layers --------------------------





