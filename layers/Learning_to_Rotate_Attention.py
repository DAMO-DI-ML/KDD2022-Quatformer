import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt, pi, log2
from utils.masking import TriangularCausalMask, ProbMask
import os
from layers.Quatformer_EncDec import TrendNorm


class QuaternionAttention(nn.Module):
    def __init__(self, query_size, key_size, mask_flag=False, scale=None, attention_dropout=0.1, output_attention=False):
        super(QuaternionAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.query_size = query_size
        self.key_size = key_size

        query_pos = (torch.arange(0.0, query_size, 1.0) / query_size).view(-1, 1, 1)
        key_pos = (torch.arange(0.0, key_size, 1.0) / key_size).view(-1, 1, 1)
        self.register_buffer('query_pos', query_pos)
        self.register_buffer('key_pos', key_pos)

    def forward(self, queries, keys, values, query_omegas, query_thetas, key_omegas, key_thetas, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        _, _, _, M = query_omegas.shape

        # a quaternion version
        Q_angles = query_omegas * self.query_pos + query_thetas # (B, L, H, M)
        K_angles = key_omegas * self.key_pos + key_thetas # (B, S, H, M)
        
        Q_cos, Q_sin = Q_angles.cos(), Q_angles.sin() # (B, L, H, M)
        K_cos, K_sin = K_angles.cos(), K_angles.sin() # (B, S, H, M)

        Q_quaternion = torch.chunk(queries, 4, dim=-1) # (B, L, H, E//4) of 4
        K_quaternion = torch.chunk(keys, 4, dim=-1) # (B, S, H, E//4) of 4
        
        Q_rotation = torch.cat(
            [
                torch.einsum('blhe,blhm->blhme', Q_quaternion[0], Q_cos) - torch.einsum('blhe,blhm->blhme', Q_quaternion[1], Q_sin),
                torch.einsum('blhe,blhm->blhme', Q_quaternion[1], Q_cos) + torch.einsum('blhe,blhm->blhme', Q_quaternion[0], Q_sin),
                torch.einsum('blhe,blhm->blhme', Q_quaternion[2], Q_cos) + torch.einsum('blhe,blhm->blhme', Q_quaternion[3], Q_sin),
                torch.einsum('blhe,blhm->blhme', Q_quaternion[3], Q_cos) - torch.einsum('blhe,blhm->blhme', Q_quaternion[2], Q_sin),
            ], dim=-1
        ) # (B, L, H, M, E)

        K_rotation = torch.cat(
            [
                torch.einsum('bshe,bshm->bshme', K_quaternion[0], K_cos) - torch.einsum('bshe,bshm->bshme', K_quaternion[2], K_sin),
                torch.einsum('bshe,bshm->bshme', K_quaternion[1], K_cos) - torch.einsum('bshe,bshm->bshme', K_quaternion[3], K_sin),
                torch.einsum('bshe,bshm->bshme', K_quaternion[2], K_cos) + torch.einsum('bshe,bshm->bshme', K_quaternion[0], K_sin),
                torch.einsum('bshe,bshm->bshme', K_quaternion[3], K_cos) + torch.einsum('bshe,bshm->bshme', K_quaternion[1], K_sin),
            ], dim=-1
        ) # (B, S, H, M, E)
        
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhme,bshme->bhls", Q_rotation, K_rotation) / M

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, query_size, key_size, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, query_omegas, query_thetas, key_omegas, key_thetas, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class LearningToRotateAttentionLayer(nn.Module):
    def __init__(self, attention, query_size, key_size, d_model, n_heads, period_type='variant', n_periods=2, d_keys=None,
                 d_values=None):

        super(LearningToRotateAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        self.register_buffer('query_D_matrix', self._gen_D_matrix(query_size))
        self.register_buffer('key_D_matrix', self._gen_D_matrix(key_size))

        kernel_size = 1
        padding = kernel_size // 2
        if period_type == 'variant':
            self.query_omega_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size, padding=padding, padding_mode='zeros')
            self.key_omega_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size, padding=padding, padding_mode='zeros')
        else:
            self.query_omega_projection = nn.Linear(d_model, n_periods * n_heads)
            self.key_omega_projection = nn.Linear(d_model, n_periods * n_heads)

        self.query_theta_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size, padding=padding, padding_mode='zeros')
        self.key_theta_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size, padding=padding, padding_mode='zeros')

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.n_heads = n_heads
        self.period_type = period_type
        self.n_periods = n_periods

    def forward(self, queries, keys, values, attn_mask=None, is_training=False):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if self.period_type == 'variant':
            query_omegas = F.relu(self.query_omega_projection(queries.transpose(1, 2))).transpose(1, 2).view(B, L, H, -1)
            key_omegas = F.relu(self.key_omega_projection(keys.transpose(1,2))).transpose(1, 2).view(B, S, H, -1)
        else:
            query_omegas = F.relu(self.query_omega_projection(torch.mean(queries, dim=1))).view(B, 1, H, -1).repeat(1, L, 1, 1)
            key_omegas = F.relu(self.key_omega_projection(torch.mean(keys, dim=1))).view(B, 1, H, -1).repeat(1, S, 1, 1)
    
        query_thetas = (F.tanh(self.query_theta_projection(queries.transpose(1, 2)).transpose(1, 2)) * pi).view(B, L, H, -1)
        key_thetas = (F.tanh(self.key_theta_projection(keys.transpose(1, 2)).transpose(1, 2)) * pi).view(B, S, H, -1)

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            query_omegas,
            query_thetas,
            key_omegas,
            key_thetas,
            attn_mask
        )
        out = out.view(B, L, -1)

        # calculate penalty of a single attention layer
        query_omegas_diff = torch.einsum('ji,bihm->bjhm', self.query_D_matrix, query_omegas)
        key_omegas_diff = torch.einsum('ji,bihm->bjhm', self.key_D_matrix, key_omegas)

        query_omegas_penalty = torch.sum(query_omegas_diff ** 2)
        key_omegas_penalty = torch.sum(key_omegas_diff ** 2)
        query_thetas_penalty = torch.sum(query_thetas ** 2)
        key_thetas_penalty = torch.sum(key_thetas ** 2)

        omegas_penalty = (query_omegas_penalty + key_omegas_penalty)
        thetas_penalty = (query_thetas_penalty + key_thetas_penalty)

        return self.out_projection(out), attn, omegas_penalty, thetas_penalty

    def _gen_D_matrix(self, L):
        """calculate the first-order difference matrix for sequence of length L
        """
        D = torch.zeros(L - 1, L)
        D[:, 1:] = torch.eye(L - 1)
        D[:, :-1] -= torch.eye(L - 1)
        return D

    def _init(self, tensor):
        dim = tensor.shape[-1]
        std = 1 / math.sqrt(dim)
        tensor.uniform_(-std, std)
        return tensor
                




class DecouplingLearningtoRotateAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, d_model, n_heads, period_type='variant', n_periods=2, d_keys=None,
                 d_values=None, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(DecouplingLearningtoRotateAttentionLayer, self).__init__()

        self.inducing_size = 96
        # self.I = nn.Parameter(torch.Tensor(1, self.inducing_size, d_model))
        # nn.init.xavier_uniform_(self.I)
        self.register_buffer('I', self._init(torch.zeros(1, self.inducing_size, d_model)))
        attn_1 = QuaternionAttention(self.inducing_size, key_size, mask_flag, scale, attention_dropout, output_attention)
        attn_2 = QuaternionAttention(query_size, self.inducing_size, mask_flag, scale, attention_dropout, output_attention)
        self.norm = TrendNorm(d_model, self.inducing_size, kernel_size=25)
        # self.decomp = series_decomp(25)
        self.attn_layer_1 = LearningToRotateAttentionLayer(attn_1, self.inducing_size, key_size, d_model, n_heads)
        self.attn_layer_2 = LearningToRotateAttentionLayer(attn_2, query_size, self.inducing_size, d_model, n_heads)

    def forward(self, queries, keys, values, attn_mask=None, is_training=False):
        inducings = self.I.repeat(queries.size(0), 1, 1)
        inducings, _, omegas_penalty_1, thetas_penalty_1 = self.attn_layer_1(inducings, keys, values)
        inducings = self.norm(inducings)
        # inducings, _ = self.decomp(inducings)
        if is_training:
            I_new = inducings.detach().mean(0, keepdim=True)
            self.I = (1 - 1e-4) * self.I + 1e-4 * I_new
        out, _, omegas_penalty_2, thetas_penalty_2 = self.attn_layer_2(queries, inducings, inducings)
        omegas_penalty = omegas_penalty_1 + omegas_penalty_2
        thetas_penalty = thetas_penalty_1 + thetas_penalty_2
        return out, None, omegas_penalty, thetas_penalty

    def _init(self, tensor):
        dim = tensor.shape[-1]
        std = 1 / math.sqrt(dim)
        tensor.uniform_(-std, std)
        return tensor