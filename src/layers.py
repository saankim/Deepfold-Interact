# %%
import torch
import torch.nn as nn
import random
import math


def gaussian_focus(distances, shift, width):
    width = width.clamp(min=5e-1)
    return torch.exp(-((distances - shift) ** 2) / width)


def laplacian_focus(distances, shift, width):
    width = width.clamp(min=5e-1)
    return torch.exp(-torch.abs(distances - shift) / width)


def cauchy_focus(distances, shift, width):
    width = width.clamp(min=5e-1)
    return 1 / (1 + ((distances - shift) / width) ** 2)


def sigmoid_focus(distances, shift, width):
    width = width.clamp(min=5e-1)
    return 1 / (1 + torch.exp((-distances + shift) / width))


def triangle_focus(distances, shift, width):
    width = width.clamp(min=5e-1)
    return torch.clamp(1 - torch.abs(distances - shift) / width, min=0)


def get_moire_focus(attention_type):
    if attention_type == "gaussian":
        return gaussian_focus
    elif attention_type == "laplacian":
        return laplacian_focus
    elif attention_type == "cauchy":
        return cauchy_focus
    elif attention_type == "sigmoid":
        return sigmoid_focus
    elif attention_type == "triangle":
        return triangle_focus
    else:
        raise ValueError("Invalid attention type")


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class Dropout(nn.Module):
    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            return nn.functional.dropout(x, p=self.p, training=self.training)
        return x


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            GaussianNoise(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.ffn(x)


class MoireAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        initial_shifts,
        initial_widths,
        focus = gaussian_focus,
    ):
        super(MoireAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert (
            self.head_dim * num_heads == output_dim
        ), "output_dim must be divisible by num_heads"
        self.focus = focus
        self.shifts = nn.Parameter(
            torch.tensor(initial_shifts, dtype=torch.float).view(1, num_heads, 1, 1)
        )
        self.widths = nn.Parameter(
            torch.tensor(initial_widths, dtype=torch.float).view(1, num_heads, 1, 1)
        )
        self.self_loop_W = nn.Parameter(
            torch.tensor(
                [1 / self.head_dim + random.uniform(0, 1) for _ in range(num_heads)],
                dtype=torch.float,
            ).view(1, num_heads, 1, 1),
            requires_grad=False,
        )
        self.qkv_proj = nn.Linear(input_dim, 3 * output_dim)
        self.scale2 = math.sqrt(self.head_dim)

    def forward(self, x, adj, mask):
        batch_size, num_nodes, _ = x.size()
        qkv = (
            self.qkv_proj(x)
            .view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale2
        moire_adj = self.focus(adj.unsqueeze(1), self.shifts, self.widths).clamp(
            min=1e-6
        )
        adjusted_scores = scores + torch.log(moire_adj)
        I = torch.eye(num_nodes, device=x.device).unsqueeze(0).unsqueeze(0)
        adjusted_scores.add_(I * self.self_loop_W)
        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
            adjusted_scores.masked_fill_(~mask_2d.unsqueeze(1), -1e6)
        attention_weights = torch.softmax(adjusted_scores, dim=-1)
        return (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .reshape(batch_size, num_nodes, -1)
        )


class MoireLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        shift_min,
        shift_max,
        dropout,
        focus=gaussian_focus,
    ):
        super(MoireLayer, self).__init__()
        shifts = [
            shift_min + random.uniform(0, 1) * (shift_max - shift_min)
            for _ in range(num_heads)
        ]
        widths = [1.3**shift for shift in shifts]
        self.attention = MoireAttention(
            input_dim,
            output_dim,
            num_heads,
            shifts,
            widths,
            focus,
        )
        self.ffn = FFN(output_dim, output_dim, output_dim, dropout)
        self.projection_for_residual = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj, mask):
        h = self.attention(x, adj, mask)
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))
        h = self.ffn(h)
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))
        x_proj = self.projection_for_residual(x)
        h = h * 0.5 + x_proj * 0.5
        return h

class LigandProjection(nn.Module):
    def __init__(self):
        super(LigandProjection, self).__init__()
        self.ffn1 = FFN(768, 384, 192)
        self.ffn2 = FFN(192, 192, 192)
        self.ffn3 = FFN(192, 96, 48)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.ffn2(x)
        x = self.ffn3(x)
        return x
    
class ProteinProjection(nn.Module):
    def __init__(self):
        super(ProteinProjection, self).__init__()
        self.soft1 = MoireLayer(1536, 768, 128, 7)
        self.soft2 = MoireLayer(768, 384, 64, 5)
        self.soft3 = MoireLayer(384, 192, 32, 4)
        self.soft4 = MoireLayer(192, 96, 32, 3)
        self.soft5 = MoireLayer(96, 48, 16, 3)

    def forward(self, x, adj):
        x = self.soft1(x, adj)
        x = self.soft2(x, adj)
        x = self.soft3(x, adj)
        x = self.soft4(x, adj)
        x = self.soft5(x, adj)
        return x
# %%
