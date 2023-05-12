import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, TokenEmbedding
from layers.Learning_to_Rotate_Attention import QuaternionAttention, LearningToRotateAttentionLayer
from layers.Quatformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, series_decomp, TrendNorm
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.lambda_1 = configs.lambda_1
        self.lambda_2 = configs.lambda_2
        self.output_attention = configs.output_attention
        self.trend_norm_enc = TrendNorm(configs.d_model, self.seq_len, kernel_size=configs.moving_avg)
        self.trend_norm_dec = TrendNorm(configs.d_model, self.label_len + self.pred_len, kernel_size=configs.moving_avg)
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        Attention = QuaternionAttention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    LearningToRotateAttentionLayer(
                        Attention(self.seq_len, self.seq_len, False, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        self.seq_len, self.seq_len, configs.d_model, configs.n_heads, period_type=configs.period_type, n_periods=configs.n_periods),
                    d_model = configs.d_model,
                    seq_len = self.seq_len,
                    d_ff = configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            # norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    LearningToRotateAttentionLayer(
                        Attention(self.label_len + self.pred_len, self.label_len + self.pred_len, False, attention_dropout=configs.dropout,
                                        output_attention=False),
                        self.label_len + self.pred_len, self.label_len + self.pred_len, configs.d_model, configs.n_heads, period_type=configs.period_type, n_periods=configs.n_periods),
                    LearningToRotateAttentionLayer(
                        Attention(self.label_len + self.pred_len, self.seq_len, False, attention_dropout=configs.dropout,
                                        output_attention=False),
                        self.label_len + self.pred_len, self.seq_len, configs.d_model, configs.n_heads, period_type=configs.period_type, n_periods=configs.n_periods),
                    configs.d_model,
                    configs.c_out,
                    self.label_len + self.pred_len,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            # norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
            # trend_projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, is_training=False):
        
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(self.device)

        # decoder input
        x_dec = torch.cat([x_enc[:, -self.label_len:, :], mean], dim=1)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.trend_norm_enc(enc_out)
        enc_out, attns, omegas_penalties_enc, thetas_penalties_enc = self.encoder(enc_out, attn_mask=enc_self_mask, is_training=is_training)

        # dec
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.trend_norm_dec(dec_out)
        dec_out, omegas_penalties_dec, thetas_penalties_dec = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, is_training=is_training)

        # final
        dec_out = dec_out + x_dec

        # regularization loss of omegas and thetas
        omegas_loss, thetas_loss = torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device)
        omegas_penalties = omegas_penalties_enc + omegas_penalties_dec
        thetas_penalties = thetas_penalties_enc + thetas_penalties_dec
        for omegas_penalty, thetas_penalty in zip(omegas_penalties, thetas_penalties):
            omegas_loss += omegas_penalty
            thetas_loss += thetas_penalty
        enc_size = self.configs.seq_len
        dec_size = self.configs.label_len + self.configs.pred_len

        omegas_loss /= (self.configs.e_layers * (2 * (enc_size - 1)) + self.configs.d_layers * ((enc_size - 1) + 3 * (dec_size - 1))) * self.configs.n_heads * self.configs.n_periods * self.configs.batch_size
        thetas_loss /= (self.configs.e_layers * (2 * enc_size) + self.configs.d_layers * (enc_size + 3 * dec_size)) * self.configs.n_heads * self.configs.n_periods * self.configs.batch_size
        regularization_loss = self.lambda_1 * omegas_loss + self.lambda_2 * thetas_loss
        # regularization_loss = omegas_loss + thetas_loss
        # regularization_loss = torch.tensor(0.).to(self.device)
        
        # output shape: [B, L, D]
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns, regularization_loss
        else:
            return dec_out[:, -self.pred_len:, :], None, regularization_loss
