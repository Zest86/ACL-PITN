# -*- coding:utf-8 -*-
"""
@Time: 2024/6/13 14:22
@Auth: Rui Wang
@File: TimesNet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
import time
import numpy as np
from torch.optim import Adam
epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='s'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        # Embed = nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='s', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' \
        #     else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    if top_list == 0:
        period = 1
    else:
        period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):

        # x = x.unsqueeze(1)
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            # period = period_list[i]
            period = 1
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res




class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(512, 1)  # Adjust the output dimension for regression

    def regression(self, x_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B, T, C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, 1)
        return output

    def forward(self, x_enc):
        x_enc = x_enc[:, :32]
        x_enc = x_enc.unsqueeze(1)
        return self.regression(x_enc)



class TimesNetPINN(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimesNetPINN, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(512, 64)  # Adjust the output dimension for regression
        self.decision = nn.Linear(67, 1)
    def regression(self, X):
        # embedding
        x_enc, feat_1, feat_2, feat_3 = X[:, :32], X[:, 32], X[:, 33], X[:, 34]

        x_enc = x_enc.unsqueeze(1)
        feat_1 = feat_1.unsqueeze(1)
        feat_2 = feat_2.unsqueeze(1)
        feat_3 = feat_3.unsqueeze(1)

        enc_out = self.enc_embedding(x_enc)  # [B, T, C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, 64)
        output = torch.cat((output, feat_1, feat_2, feat_3), 1)
        output = self.decision(output)
        return output

    def regression_net(self, x_enc, feat_1, feat_2, feat_3):
        # embedding
        return self.regression(torch.cat((x_enc, feat_1, feat_2, feat_3), 1))

    def Physics_net(self, feature, feat_1, feat_2, feat_3):
        # No need to calculate the gradient of adversarial loss
        u = self.regression_net(feature, feat_1, feat_2, feat_3)
        u_feat_1 = torch.autograd.grad(u, feat_1,
                                       grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_feat_2 = torch.autograd.grad(u, feat_2,
                                       grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_feat_3 = torch.autograd.grad(u, feat_3,
                                       grad_outputs=torch.ones_like(u), create_graph=True)[0]
        pred_physics = (u[:-1, 0]
                        + (u_feat_1[:-1, 0] * (feat_1[1:, 0] - feat_1[:-1, 0]))
                        + (u_feat_2[:-1, 0] * (feat_2[1:, 0] - feat_2[:-1, 0]))
                        + (u_feat_3[:-1, 0] * (feat_3[1:, 0] - feat_3[:-1, 0]))
                        )
        return u, pred_physics

    def forward(self, X, flag):
        # flag to show calculate physics loss or not
        self.feature = X[:, :32].clone().detach().requires_grad_(True)
        self.feat_1 = X[:, 32].clone().detach().requires_grad_(True).unsqueeze(1)
        self.feat_2 = X[:, 33].clone().detach().requires_grad_(True).unsqueeze(1)
        self.feat_3 = X[:, 34].clone().detach().requires_grad_(True).unsqueeze(1)

        if flag:
            u, pred_physics = self.Physics_net(self.feature, self.feat_1, self.feat_2, self.feat_3)
            return u, pred_physics
        else:
            u = self.regression(X)
            return u



def train_Net(inp_comb, train_out, inp_comb_test, test_out, subject_id, BP_type):
    # initialize the model, using TimesNet
    # model_dnn_conv = DNNModel(np.shape(train_beats)[-1], 1, 64)
    configs = Configs()

    # model = iTransformer(configs)
    model = TimesNet(configs)
    model.to(device)
    # inp_comb, train_out = inp_comb.to(device), train_out.to(device)
    # optimizer = optim.Adam(model_dnn_conv.parameters(), lr=3e-4)

    # initialize the model, using RNN

    inp_comb = torch.tensor(inp_comb, dtype=torch.float32)
    train_out = torch.tensor(train_out, dtype=torch.float32)
    inp_comb, train_out = inp_comb.to(device), train_out.to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)

    # Two lists are initialized to keep track of the training and testing loss during each epoch
    loss_list_conv = []
    test_loss_list_conv = []
    best_loss = float('inf')
    print("TimesNet model training started")
    for epoch in range(1000):
        start = time.time()
        train_l_sum = 0.0
        output = model(inp_comb)
        optimizer.zero_grad()
        loss = output - train_out
        loss = torch.mean(torch.square(loss))
        loss.backward()
        optimizer.step()

        # model evaluation
        # model_dnn_conv.eval()
        # for X, y in test_data_iter:
        #     output = model_dnn_conv(X)
        #     loss = criterion(output, y)
        #     loss_list_conv.append(float(loss.item()))

        # test phase
        inp_comb_test, test_out = inp_comb_test.to(device=device), test_out.to(device=device)
        pred_out = model(inp_comb_test)
        test_loss = pred_out - test_out
        test_loss = torch.mean(torch.square(test_loss))
        end = time.time()
        if epoch % 100 == 0:
            print("epoch: {}, time: {}, train_loss: {}, test_loss: {}".format(epoch, round(end - start, 2), round(loss.item(), 2), round(test_loss.item(), 2)))

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "ringCPT/content/TimesNet/" + subject_id + "/" + BP_type + "/best.pth")


        # If the training loss reaches a minimum value of 0.01, or the maximum number of epochs is reached, the training process is stopped
        if (loss.item() <= 0.01) | (epoch == epochs - 1):
            torch.save(model.state_dict(), "ringCPT/content/TimesNet/" + subject_id + "/" + BP_type + "/last.pth")
            print("TimesNet model training Completed. Epoch %d/%d -- loss: %.4f" % (epoch, epochs, float(loss)))
            # np.save("results/PINN/loss_list_conv.npy", loss_list_conv)
            # np.save("test_loss_list_conv.npy", test_loss_list_conv)
            return loss_list_conv, test_loss_list_conv




class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

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

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder

        if self.task_name == 'regression':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model, 1)


    def regression(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc):
        x = x_enc[:, :32]
        x = x.unsqueeze(1)
        if self.task_name == 'regression':
            dec_out = self.regression(x)
            return dec_out  # [B, N]
        return None


class Configs:
    def __init__(self):
        # General parameters
        self.task_name = 'regression'  # or 'short_term_forecast', 'imputation', 'anomaly_detection', 'regression'
        self.seq_len = 32  # Input sequence length
        self.label_len = 48  # Label length for forecasting
        self.pred_len = 24  # Prediction length
        self.e_layers = 2  # Number of encoder layers
        self.d_model = 512  # Dimension of the model
        self.embed = 'timeF'  # Embedding type, e.g., 'timeF' for time features
        self.freq = 's'  # Frequency of the data, e.g., 'h' for hourly data
        self.dropout = 0.1  # Dropout rate
        self.top_k = 1
        self.d_ff = 512  # Dimension of the feedforward network model
        self.num_kernels = 6 # Number of kernels in the Inception block
        self.enc_in = 32  # Number of input features (for forecasting and imputation)
        self.output_attention = False  # Whether to output attention weights
        self.factor = 1 # 'attn factor'
        self.n_heads = 8  # Number of heads in the multi-head attention
        self.activation = 'gelu'  # Activation function
        # Task-specific parameters
        # self.enc_in = 7  # Number of input features (for forecasting and imputation)
        # self.c_out = 7  # Number of output features (for forecasting and imputation)
        # self.num_class = 10  # Number of classes (for classification)


def calculate_performance_Net(inp_comb_test, test_out, scaler_out, subject, BP_type):
    # The trained model's predictions on the test dataset are computed

    configs = Configs()
    model = TimesNet(configs)
    # use PINN_BN for full model
    model.load_state_dict(
        torch.load("ringCPT/content/TimesNet/" + subject + "/" + BP_type + "/best.pth"))
    test_out = test_out.tolist()
    pred_out = model(inp_comb_test)
    pred_out = pred_out.tolist()
    corr = np.corrcoef(np.concatenate(test_out)[:], np.concatenate(pred_out)[:])[0][1]
    rmse = np.sqrt(np.mean(np.square(
        np.concatenate(scaler_out.inverse_transform(np.concatenate(test_out)[:][:, None])) -
        np.concatenate(scaler_out.inverse_transform(np.concatenate(pred_out)[:][:, None])))))

    print('#### TimesNet Performance ####')
    print('subject: {}, BP_type: {}'.format(subject, BP_type))
    print('Corr: %.2f,  RMSE: %.1f' % (corr, rmse))
    print('----------------------------------')