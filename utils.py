# -*- coding:utf-8 -*-
"""
@Time: 2024/7/10 15:20
@Auth: Rui Wang
@File: models.py
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import torch.optim as optim
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


class TemporalBlock(nn.Module):
    def __init__(self, configs):
        super(TemporalBlock, self).__init__()
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


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='s', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)


def training_data_latin_hypercube(X, u, N_inner=1e3):

    # Latin Hypercube sampling for collocation points
    lb = torch.min(X[:, :32], dim=1)[0]
    lb = torch.min(lb)
    ub = torch.max(X[:, :32], dim=1)[0]
    ub = torch.max(ub)



    x_train = X
    x_boundary = torch.tensor([lb, ub]).reshape(2, 1).to(X.device)
    # create boundary for regression model
    # u_boundary = np.array([0, 1]).reshape(2, 1)
    u_lb = torch.min(u)
    u_ub = torch.max(u)
    u_boundary = torch.tensor([u_lb, u_ub]).reshape(2, 1).to(X.device)
    return x_train, x_boundary, u_boundary


def generate_attack_samples(model, device, x_train, N0, train_out, n_samples=300, lb=[0], ub=[1], steps=2, eps=2e-1,
                            eta=2e-2, m=0):

    attack = regression_PGD(model, lb=lb, ub=ub, steps=steps, eps=eps, eta=eta)
    x_adv = attack.attack(x_train, train_out).cpu().detach().numpy()
    x_adv[:, 32:] = x_train[:, 32:]
    return x_adv


class regression_PGD:
    def __init__(self, model, lb, ub, eps=0.1, eta=0.02, steps=20, loss=nn.L1Loss()):
        self.model = model
        self.eps = eps
        self.eta = eta
        self.steps = steps
        self.loss = loss
        self.lb = lb
        self.ub = ub
        self.device = next(model.parameters()).device

    def attack(self, samples, train_out):
        # return adversarial samples in multi steps
        ans = None
        if torch.is_tensor(samples) != True:
            samples = torch.from_numpy(samples.copy()).float().to(self.device)
        else:
            samples = samples.clone().detach()
        adv_samples = samples.clone().detach()

        # judge if train_out is on GPU
        if train_out.device != self.device:
            train_out = train_out.to(self.device)

        adv_samples += torch.empty_like(adv_samples).uniform_(-self.eps, self.eps)
        for i in range(len(self.lb)):
            adv_samples[:, i] = torch.clamp(adv_samples[:, i], min=self.lb[i], max=self.ub[i])
        adv_samples = adv_samples.detach()

        for _ in range(self.steps):
            adv_samples.requires_grad = True
            # original model
            f = self.model.function(adv_samples, train_out)
            # cost = self.loss(f, torch.zeros(f.shape).to(self.device))

            # Update adversarial images
            grad = torch.autograd.grad(f, adv_samples, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True, allow_unused=True)[0]

            if grad is None:
                adv_samples = adv_samples.detach() + self.eta * 0
            else:
                adv_samples = adv_samples.detach() + self.eta*grad.sign()
            delta = torch.clamp(adv_samples - samples, min=-self.eps, max=self.eps)
            adv_samples = samples + delta
            for i in range(len(self.lb)):
                adv_samples[:, i] = torch.clamp(adv_samples[:, i], min=self.lb[i], max=self.ub[i])
            adv_samples = adv_samples.detach()

            # if ans is None:
            #     ans = adv_samples
            # else:
            #     ans = torch.cat((ans, adv_samples), dim=0)
        return adv_samples


class diffusion_foward():
    def __init__(self, ):
        self.n_steps = 100
        self.betas = self.make_beta_schedule(schedule='sigmoid', n_timesteps=self.n_steps, start=1e-5, end=1e-2)
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            # 判断x_0是否是tensor类，如果是numpy类的话转成tensor
            if not torch.is_tensor(x_0):
                x_0 = torch.tensor(x_0, dtype=torch.float32)
            noise = torch.randn_like(x_0)
        alphas_t = self.extract(self.alphas_bar_sqrt, t, x_0)
        alphas_1_m_t = self.extract(self.one_minus_alphas_bar_sqrt, t, x_0)
        return (alphas_t * x_0 + alphas_1_m_t * noise)


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits

def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=1.0):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def set_temperature(self, temp=1.0):
        self.temperature = temp

    def forward(self, outputs):
        feats = outputs['feats']  # feats shape: [B, D]
        labels = outputs['labels']  # labels shape: [B]

        device = (torch.device('cuda')
                  if feats.is_cuda
                  else torch.device('cpu'))

        feats = F.normalize(feats, dim=-1, p=2)
        local_batch_size = feats.size(0)

        # all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)
        # all_labels = concat_all_gather(labels)  # no gradient gather
        all_feats = feats.clone().detach()
        all_labels = labels.clone().detach()
        # compute the mask based on labels
        if local_batch_size != self.last_local_batch_size:
            mask = torch.eq(labels.view(-1, 1),
                            all_labels.contiguous().view(1, -1)).float().to(device)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(device) +
                local_batch_size * get_rank(),
                0
            )

            self.last_local_batch_size = local_batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return loss

