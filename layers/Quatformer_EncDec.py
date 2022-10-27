import torch
import torch.nn as nn
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class TrendNorm(nn.Module):

    def __init__(self,
                 dimension: int,
                 seq_len: int,
                 order: int = 1,
                 kernel_size: int = 25,
                 eps: float = 1e-5,
                 affine: bool = True):

        super(TrendNorm, self).__init__()
        if affine:
            self.gamma = nn.Parameter(torch.ones(dimension,))
            self.register_buffer(f'position', torch.arange(0.0, seq_len, 1.0) / seq_len)
            self.betas = nn.ParameterList([nn.Parameter(torch.zeros(dimension,)) for _ in range(order+1)])
        self.detrend = series_decomp(kernel_size)
        self.eps = eps
        self.order = order
        self.affine = affine
        
    def forward(self, tensor: torch.Tensor):
        B, L, E = tensor.shape
        tensor, _ = self.detrend(tensor)
        std = tensor.std(1, unbiased=False, keepdim=True)

        if self.affine:
            trend = torch.einsum('l,e->le',self.position**0, self.betas[0])
            for i in range(1,self.order+1):
                trend += torch.einsum('l,e->le',self.position**i, self.betas[i])
            trend = trend.view(1, L, E).repeat(B, 1, 1)
            return self.gamma * tensor / (std + self.eps) + trend
        else:
            return tensor / (std + self.eps)


class EncoderLayer(nn.Module):
    """
    Complexformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, seq_len, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.norm1 = TrendNorm(d_model, seq_len, kernel_size=moving_avg)
        self.norm2 = TrendNorm(d_model, seq_len, kernel_size=moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, is_training=False):
        new_x, attn, omegas_penalty, thetas_penalty = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            is_training=is_training
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res = self.norm2(x + y)
        return res, attn, omegas_penalty, thetas_penalty


class Encoder(nn.Module):
    """
    Complexformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, is_training=False):
        attns = []
        omegas_penalties = []
        thetas_penalties = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn, omegas_penalty, thetas_penalty = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
                omegas_penalties.append(omegas_penalty)
                omegas_penalties.append(thetas_penalty)
            x, attn, omegas_penalty, thetas_penalty = self.attn_layers[-1](x)
            omegas_penalties.append(omegas_penalty)
            omegas_penalties.append(thetas_penalty)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn, omegas_penalty, thetas_penalty  = attn_layer(x, attn_mask=attn_mask, is_training=False)
                attns.append(attn)
                omegas_penalties.append(omegas_penalty)
                omegas_penalties.append(thetas_penalty)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, omegas_penalties, thetas_penalties


class DecoderLayer(nn.Module):
    """
    Complexformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, seq_len, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.norm1 = TrendNorm(d_model, seq_len, kernel_size=moving_avg)
        self.norm2 = TrendNorm(d_model, seq_len, kernel_size=moving_avg)
        self.norm3 = TrendNorm(d_model, seq_len, kernel_size=moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=95, stride=1, padding=47,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, is_training=False):
        new_x_1, _, omegas_penalty_1, thetas_penalty_1 = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        x = x + self.dropout(new_x_1)
        x = self.norm1(x)

        new_x_2, _, omegas_penalty_2, thetas_penalty_2 = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            is_training=is_training
        )
        x = self.norm2(x)

        x = x + self.dropout(new_x_2)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm3(x + y)

        return x, omegas_penalty_1 + omegas_penalty_2, thetas_penalty_1 + thetas_penalty_2


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, is_training=False):
        omegas_penalties = []
        thetas_penalties = []
        for layer in self.layers:
            x, omegas_penalty, thetas_penalty = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, is_training=is_training)
            omegas_penalties.append(omegas_penalty)
            thetas_penalties.append(thetas_penalty)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, omegas_penalties, thetas_penalties
