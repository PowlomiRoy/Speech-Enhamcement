import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import functools
import os, numbers, math, signal
from typing import Union, List, Sequence
from distutils.version import LooseVersion
import warnings
import torch as th

warnings.filterwarnings("ignore")
EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)

import torch
import numpy as np
from typing import Union

class RealTensor:
    def __init__(self, real: Union[torch.Tensor, np.ndarray]):
        if isinstance(real, np.ndarray):
            real = torch.from_numpy(real)

        if not torch.is_tensor(real):
            raise TypeError(f'The input must be a torch.Tensor or np.ndarray, but got {type(real)}')

        self.real = real
    
    def __getitem__(self, item) -> 'RealTensor':
        return RealTensor(self.real[item])

    def __setitem__(self, item, value: Union['RealTensor', torch.Tensor, numbers.Number]):
        if isinstance(value, RealTensor):
            self.real[item] = value.real
        else:
            self.real[item] = value

    def __mul__(self, other: Union['RealTensor', torch.Tensor, numbers.Number]) -> 'RealTensor':
        if isinstance(other, RealTensor):
            return RealTensor(self.real * other.real)
        else:
            return RealTensor(self.real * other)

    def __rmul__(self, other: Union['RealTensor', torch.Tensor, numbers.Number]) -> 'RealTensor':
        return self * other  # Multiplication is commutative for real values

    def __imul__(self, other):
        if isinstance(other, RealTensor):
            self.real *= other.real
        else:
            self.real *= other
        return self
    
    def __truediv__(self, other) -> 'RealTensor':
        if isinstance(other, RealTensor):
            return RealTensor(self.real / other.real)
        else:
            return RealTensor(self.real / other)

    def __rtruediv__(self, other) -> 'RealTensor':
        return RealTensor(other / self.real)
    
    def conj(self) -> 'RealTensor':
        return RealTensor(self.real)
    
    def __itruediv__(self, other) -> 'RealTensor':
        if isinstance(other, RealTensor):
            self.real /= other.real
        else:
            self.real /= other
        return self
    def __add__(self, other) -> 'RealTensor':
        if isinstance(other, RealTensor):
            return RealTensor(self.real + other.real)
        else:
            return RealTensor(self.real + other)

    def __radd__(self, other) -> 'RealTensor':
        return RealTensor(other + self.real)

    def __iadd__(self, other) -> 'RealTensor':
        if isinstance(other, RealTensor):
            self.real += other.real
        else:
            self.real += other
        return self
    
    def __sub__(self, other) -> 'RealTensor':
        if isinstance(other, RealTensor):
            return RealTensor(self.real - other.real)
        else:
            return RealTensor(self.real - other)

    def __rsub__(self, other) -> 'RealTensor':
        return RealTensor(other - self.real)

    def __isub__(self, other) -> 'RealTensor':
        if isinstance(other, RealTensor):
            self.real -= other.real
        else:
            self.real -= other
        return self
    def __matmul__(self, other) -> 'RealTensor':
        if isinstance(other, RealTensor):
            o_real = torch.matmul(self.real, other.real)
        else:
            o_real = torch.matmul(self.real, other)
        return RealTensor(o_real)

    def __rmatmul__(self, other) -> 'RealTensor':
        if isinstance(other, RealTensor):
            o_real = torch.matmul(other.real, self.real)
        else:
            o_real = torch.matmul(other, self.real)
        return RealTensor(o_real)

    def __imatmul__(self, other) -> 'RealTensor':
        if isinstance(other, RealTensor):
            self.real = torch.matmul(self.real, other.real)
        else:
            self.real @= other
        return self
    
    def __neg__(self) -> 'RealTensor':
        return RealTensor(-self.real)

    def __eq__(self, other) -> torch.Tensor:
        if isinstance(other, RealTensor):
            return self.real == other.real
        else:
            return self.real == other

    def __len__(self) -> int:
        return len(self.real)

    def __repr__(self) -> str:
        return 'RealTensor(\nReal:\n' + repr(self.real) + '\n)'

    def __abs__(self) -> torch.Tensor:
        return self.real.abs()

    def __pow__(self, exponent) -> 'RealTensor':
        if exponent == -2:
            return 1 / (self * self)
        if exponent == -1:
            return 1 / self
        if exponent == 0:
            return RealTensor(torch.ones_like(self.real))
        if exponent == 1:
            return self.clone()
        if exponent == 2:
            return self * self
        return RealTensor(self.real.pow(exponent))

    def __ipow__(self, exponent) -> 'RealTensor':
        self.real = self.real.pow(exponent)
        return self

    def abs(self) -> torch.Tensor:
        return self.real.abs()

    def backward(self) -> None:
        self.real.backward()

    def byte(self) -> 'RealTensor':
        return RealTensor(self.real.byte())

    def clone(self) -> 'RealTensor':
        return RealTensor(self.real.clone())

    def flatten(self, dim) -> 'RealTensor':
        return RealTensor(torch.flatten(self.real, dim))

    def reshape(self, *shape) -> 'RealTensor':
        return RealTensor(self.real.reshape(*shape))

    def zeromean(self, dim) -> 'RealTensor':
        self.real = self.real - torch.mean(self.real, dim=dim, keepdim=True)
        return self

# The following are complex-only and now removed entirely:
# - angle
# - conj
# - conj_
    def contiguous(self) -> 'RealTensor':
        return RealTensor(self.real.contiguous())

    def copy_(self) -> 'RealTensor':
        self.real = self.real.copy_()
        return self

    def cpu(self) -> 'RealTensor':
        return RealTensor(self.real.cpu())

    def cuda(self) -> 'RealTensor':
        return RealTensor(self.real.cuda())

    def expand(self, *sizes) -> 'RealTensor':
        return RealTensor(self.real.expand(*sizes))

    def expand_as(self, *args, **kwargs) -> 'RealTensor':
        return RealTensor(self.real.expand_as(*args, **kwargs))

    def detach(self) -> 'RealTensor':
        return RealTensor(self.real.detach())

    def detach_(self) -> 'RealTensor':
        self.real.detach_()
        return self

    @property
    def device(self):
        return self.real.device

    def diag(self) -> 'RealTensor':
        return RealTensor(self.real.diag())

    def diagonal(self) -> 'RealTensor':
        return RealTensor(self.real.diag())

    def dim(self) -> int:
        return self.real.dim()

    def double(self) -> 'RealTensor':
        return RealTensor(self.real.double())

    @property
    def dtype(self) -> torch.dtype:
        return self.real.dtype

    def eq(self, other) -> torch.Tensor:
        if isinstance(other, RealTensor):
            return self.real == other.real
        else:
            return self.real == other

    def equal(self, other) -> bool:
        if isinstance(other, RealTensor):
            return self.real.equal(other.real)
        else:
            return self.real.equal(other)

    def float(self) -> 'RealTensor':
        return RealTensor(self.real.float())

    def fill(self, value) -> 'RealTensor':
        return RealTensor(self.real.fill(value))

    def fill_(self, value) -> 'RealTensor':
        self.real.fill_(value)
        return self

    def gather(self, dim, index) -> 'RealTensor':
        return RealTensor(self.real.gather(dim, index))

    def get_device(self, *args, **kwargs):
        return self.real.get_device(*args, **kwargs)

    def half(self) -> 'RealTensor':
        return RealTensor(self.real.half())

    def index_add(self, dim, index, tensor) -> 'RealTensor':
        return RealTensor(self.real.index_add(dim, index, tensor))

    def index_copy(self, dim, index, tensor) -> 'RealTensor':
        return RealTensor(self.real.index_copy(dim, index, tensor))

    def index_fill(self, dim, index, value) -> 'RealTensor':
        return RealTensor(self.real.index_fill(dim, index, value))

    def index_select(self, dim, index) -> 'RealTensor':
        return RealTensor(self.real.index_select(dim, index))
    
    def inverse(self):
    # m x n x n
        in_size = self.size()
        a = self.view(-1, self.size(-1), self.size(-1))

        try:
            inv = a.inverse()
        except Exception:
            raise

        return RealTensor(inv.view(*in_size))

    def item(self) -> numbers.Number:
        return self.real.item()

    def masked_fill(self, mask, value) -> 'RealTensor':
        return RealTensor(self.real.masked_fill(mask, value))

    def masked_fill_(self, mask, value) -> 'RealTensor':
        self.real.masked_fill_(mask, value)
        return self

    def mean(self, *args, **kwargs) -> 'RealTensor':
        return RealTensor(self.real.mean(*args, **kwargs))

    def neg(self) -> 'RealTensor':
        return RealTensor(-self.real)

    def neg_(self) -> 'RealTensor':
        self.real.neg_()
        return self

    def nelement(self) -> int:
        return self.real.nelement()

    def numel(self) -> int:
        return self.real.numel()
    def new(self, *args, **kwargs) -> 'RealTensor':
        return RealTensor(self.real.new(*args, **kwargs))

    def new_empty(self, size, dtype=None, device=None, requires_grad=False) -> 'RealTensor':
        real = self.real.new_empty(size,
                                   dtype=dtype,
                                   device=device,
                                   requires_grad=requires_grad)
        return RealTensor(real)

    def new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False) -> 'RealTensor':
        if isinstance(fill_value, complex):
            real_value = fill_value.real
        else:
            real_value = fill_value

        real = self.real.new_full(size,
                                  fill_value=real_value,
                                  dtype=dtype,
                                  device=device,
                                  requires_grad=requires_grad)
        return RealTensor(real)

    def new_tensor(self, data, dtype=None, device=None, requires_grad=False) -> 'RealTensor':
        if isinstance(data, RealTensor):
            real = data.real
        elif isinstance(data, np.ndarray):
            if data.dtype.kind == 'c':
                real = data.real
            else:
                real = data
        else:
            real = data

        real = self.real.new_tensor(real,
                                    dtype=dtype,
                                    device=device,
                                    requires_grad=requires_grad)
        return RealTensor(real)

    def numpy(self) -> np.ndarray:
        return self.real.numpy()

    def permute(self, *dims) -> 'RealTensor':
        return RealTensor(self.real.permute(*dims))

    def pow(self, exponent) -> 'RealTensor':
        return self ** exponent

    def requires_grad_(self) -> 'RealTensor':
        self.real.requires_grad_()
        return self

    @property
    def requires_grad(self):
        return self.real.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
            self.real.requires_grad = value

    def repeat(self, *sizes):
        return RealTensor(self.real.repeat(*sizes))

    def retain_grad(self) -> 'RealTensor':
        self.real.retain_grad()
        return self

    def share_memory_(self) -> 'RealTensor':
        self.real.share_memory_()
        return self

    @property
    def shape(self) -> torch.Size:
        return self.real.shape

    def size(self, *args, **kwargs) -> torch.Size:
        return self.real.size(*args, **kwargs)

    def sqrt(self) -> 'RealTensor':
        return self ** 0.5

    def squeeze(self, dim) -> 'RealTensor':
        return RealTensor(self.real.squeeze(dim))

    def sum(self, *args, **kwargs) -> 'RealTensor':
        return RealTensor(self.real.sum(*args, **kwargs))

    def take(self, indices) -> 'RealTensor':
        return RealTensor(self.real.take(indices))

    def to(self, *args, **kwargs) -> 'RealTensor':
        return RealTensor(self.real.to(*args, **kwargs))

    def tolist(self) -> List[numbers.Number]:
        return self.real.tolist()

    def transpose(self, dim0, dim1) -> 'RealTensor':
        return RealTensor(self.real.transpose(dim0, dim1))

    def transpose_(self, dim0, dim1) -> 'RealTensor':
        self.real.transpose_(dim0, dim1)
        return self

    def type(self) -> str:
        return self.real.type()

    def unfold(self, dim, size, step):
        return RealTensor(self.real.unfold(dim, size, step))

    def unsqueeze(self, dim) -> 'RealTensor':
        return RealTensor(self.real.unsqueeze(dim))

    def unsqueeze_(self, dim) -> 'RealTensor':
        self.real.unsqueeze_(dim)
        return self

    def view(self, *args, **kwargs) -> 'RealTensor':
        return RealTensor(self.real.view(*args, **kwargs))

    def view_as(self, tensor):
        return self.view(tensor.size())
    
def init_kernel(frame_len, frame_hop, num_fft=None, window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    fft_size = 2 ** math.ceil(math.log2(frame_len)) if not num_fft else num_fft
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    w = torch.fft.rfft(torch.eye(fft_size) / S_)
    kernel = torch.stack([w.real, w.imag], -1)
    kernel = torch.transpose(kernel, 0, 2) * window
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class STFTBase(nn.Module):
    def __init__(self, frame_len, frame_hop, window="sqrt_hann", num_fft=None):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len, frame_hop, num_fft=num_fft, window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self): self.K.requires_grad = False

    def unfreeze(self): self.K.requires_grad = True

    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))

    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(self.window, self.stride,
                                                                                             self.K.requires_grad,
                                                                                             self.K.shape)
    
class STFT(STFTBase):
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            print(x.shape)
            raise RuntimeError(f"Expect 2D/3D tensor, but got {x.dim()}D")
        self.check_nan()

        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p, r, i
    
class iSTFT(STFTBase):
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p=None, squeeze=False):
        self.check_nan()

        if p is None:
            # Extract underlying tensor if m is a RealTensor
            base_tensor = m.real if hasattr(m, 'real') else m
            p = torch.zeros_like(base_tensor)

        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(p.dim()))

        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)

        base_tensor = m.real if hasattr(m, 'real') else m
        r = base_tensor * torch.cos(p)
        i = base_tensor * torch.sin(p)
        c = torch.cat([r, i], dim=1)

        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)

        if squeeze:
            s = torch.squeeze(s)
        return s


    
class Conv1D(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError(f"{self.__class__.__name__} accepts 2D or 3D tensor as input")
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self._name = 'Identity'

    def forward(self, x): return x

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
    
class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x
    
class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y
    
def _freal(func, nthargs=0):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> torch.Tensor:
        return func(*args, **kwargs)
    return wrapper

def einsum(equation, operands):
    """Einsum for real-valued tensors only."""
    real_operands = [x.real if isinstance(x, RealTensor) else x for x in operands]
    result = torch.einsum(equation, real_operands)
    return RealTensor(result)


def cat(seq: Sequence[torch.Tensor], dim=0, out=None):
    """
    Concatenate real tensors

    Args:
        seq: list of torch.Tensor
        dim: dimension along which to concatenate
        out: optional output tensor
    Returns:
        torch.Tensor
    """
    return torch.cat(seq, dim=dim, out=out)

def stack(seq: Sequence[torch.Tensor], dim=0, out=None):
    """
    Stack a sequence of real tensors along a new dimension.

    Args:
        seq: list of torch.Tensor
        dim: dimension to stack along
        out: optional output tensor
    Returns:
        torch.Tensor
    """
    return torch.stack(seq, dim=dim, out=out)

def reverse(tensor: torch.Tensor, dim=0) -> torch.Tensor:
    """
    Reverse a real tensor along a specified dimension.

    Args:
        tensor: torch.Tensor
        dim: dimension to reverse
    Returns:
        torch.Tensor
    """
    idx = torch.arange(tensor.size(dim) - 1, -1, -1, device=tensor.device)
    return tensor.index_select(dim, idx)

def signal_frame(signal: torch.Tensor,
                 frame_length: int, frame_step: int,
                 pad_value=0) -> torch.Tensor:
    """
    Expands real-valued signal into frames of given length.

    Args:
        signal: (B * F, D, T)
        frame_length: int
        frame_step: int
        pad_value: padding value
    Returns:
        torch.Tensor: (B * F, D, T, W)
    """
    signal = F.pad(signal, (0, frame_length - 1), 'constant', pad_value)
    indices = sum([list(range(i, i + frame_length))
                   for i in range(0, signal.size(-1) - frame_length + 1,
                                  frame_step)], [])
    signal = signal[..., indices].view(*signal.size()[:-1], -1, frame_length)
    return signal

def trace(a: torch.Tensor) -> torch.Tensor:
    """
    Compute the trace of a batch of real square matrices.

    Args:
        a: Tensor of shape (..., N, N)
    Returns:
        Tensor of shape (...), sum of diagonals
    """
    E = torch.eye(a.size(-1), dtype=torch.bool, device=a.device).expand_as(a)
    return a[E].view(*a.shape[:-2], a.shape[-1]).sum(-1)

def allclose(a: torch.Tensor,
             b: torch.Tensor,
             rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    """
    Check if two real-valued tensors are approximately equal.

    Args:
        a: torch.Tensor
        b: torch.Tensor
        rtol: relative tolerance
        atol: absolute tolerance
        equal_nan: whether to compare NaNs as equal
    Returns:
        bool
    """
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication for real-valued tensors.

    Args:
        a: torch.Tensor
        b: torch.Tensor
    Returns:
        torch.Tensor
    """
    return torch.matmul(a, b)

def real_matrix2real_matrix(c: torch.Tensor) -> torch.Tensor:
    assert c.size(-2) == c.size(-1), c.size()
    # (∗, m, m) -> (*, 2m, 2m)
    return torch.cat(
        [torch.cat([c, -c], dim=-1), torch.cat([c, c], dim=-1)],
        dim=-2,
    )


def real_vector2real_vector(c: torch.Tensor) -> torch.Tensor:
    # (∗, m, k) -> (*, 2m, k)
    return torch.cat([c, c], dim=-2)


def real_matrix2real_matrix(c: torch.Tensor) -> torch.Tensor:
    assert c.size(-2) == c.size(-1), c.size()
    # (∗, 2m, 2m) -> (*, m, m)
    n = c.size(-1)
    assert n % 2 == 0, n
    real = c[..., : n // 2, : n // 2]
    return real


def real_vector2real_vector(c: torch.Tensor) -> torch.Tensor:
    # (∗, 2m, k) -> (*, m, k)
    n = c.size(-2)
    assert n % 2 == 0, n
    real = c[..., : n // 2, :]
    return real


def solve(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Solve ax = b"""
    a = real_matrix2real_matrix(a)
    b = real_vector2real_vector(b)
    x, LU = torch.solve(b, a)

    return real_vector2real_vector(x), real_matrix2real_matrix(LU)
