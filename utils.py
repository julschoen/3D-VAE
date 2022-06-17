from itertools import chain
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributed as dist

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Quantizer(nn.Module):
    # Code taken from:
    # https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I

    """
    EMA-updated Vector Quantizer
    """
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, laplace_alpha=1e-5
    ):
        super(Quantizer, self).__init__()

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed) # e_i
        self.register_buffer("embed_avg", embed.clone()) # m_i
        self.register_buffer("cluster_size", torch.zeros(num_embeddings)) # N_i
        self.register_buffer("first_pass", torch.as_tensor(1))

        self.commitment_cost = commitment_cost

        self.decay = decay
        self.laplace_alpha = laplace_alpha

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

    def embed_code(self, embed_idx):
        return F.embedding(embed_idx, self.embed)

    def _update_ema(self, flat_input, encoding_indices):
        # buffer updates need to be in-place because of distributed
        encodings_one_hot = F.one_hot(
            encoding_indices, num_classes=self.num_embeddings
        ).type_as(flat_input)

        new_cluster_size = encodings_one_hot.sum(dim=0)
        dw = encodings_one_hot.T @ flat_input

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(new_cluster_size)
            torch.distributed.all_reduce(dw)

        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=(1-self.decay)
        )

        self.embed_avg.data.mul_(self.decay).add_(dw, alpha=(1-self.decay))

        # Laplacian smoothing
        n = self.cluster_size.sum()
        cluster_size = n * ( # times n because we don't want probabilities but counts
            (self.cluster_size + self.laplace_alpha)
            / (n + self.num_embeddings * self.laplace_alpha)
        )

        embed_normalized = self.embed_avg / cluster_size.unsqueeze(dim=-1)
        self.embed.data.copy_(embed_normalized)

    def _init_ema(self, flat_input):
        mean = flat_input.mean(dim=0)
        std = flat_input.std(dim=0)
        cluster_size = flat_input.size(dim=0)

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(mean)
            torch.distributed.all_reduce(std)
            mean /= torch.distributed.get_world_size()
            std /= torch.distributed.get_world_size()

            cluster_size *= torch.distributed.get_world_size()

        self.embed.mul_(std)
        self.embed.add_(mean)
        self.embed_avg.copy_(self.embed)

        self.cluster_size.data.add_(cluster_size / self.num_embeddings)
        self.first_pass.mul_(0)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, inputs):
        inputs = inputs.float()
        with torch.no_grad():
            channel_last = inputs.permute(0, 2, 3, 4, 1) # XXX: might not actually be necessary
            input_shape = channel_last.shape

            flat_input = channel_last.reshape(-1, self.embedding_dim)

            if self.training and self.first_pass:
                self._init_ema(flat_input)

            encoding_indices = torch.argmin(
                torch.cdist(flat_input, self.embed, compute_mode='donot_use_mm_for_euclid_dist')
            , dim=1)
            quantized = self.embed_code(encoding_indices).reshape(input_shape)

            if self.training:
                self._update_ema(flat_input, encoding_indices)

            # Cast everything back to the same order and dimensions of the input
            quantized = quantized.permute(0, 4, 1, 2, 3)
            encoding_indices = encoding_indices.reshape(input_shape[:-1])

        # Don't need to detach quantized; doesn't require grad
        e_latent_loss = F.mse_loss(quantized, inputs)
        loss = self.commitment_cost * e_latent_loss

        # Trick to have identity backprop grads
        quantized = inputs + (quantized - inputs).detach()

        # don't change this order without checking everything
        return (loss, quantized, encoding_indices)

class Encoder(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        n_res_block,
        n_res_channel,
        stride=2
    ):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv3d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv3d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, data):
        return self.blocks(data)

class Decoder(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        channel,
        n_res_block,
        n_res_channel,
        stride=2
    ):
        super().__init__()

        blocks = [nn.Conv3d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose3d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose3d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class MyQuantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def get_world_size(self):
        if not dist.is_available():
            return 1

        if not dist.is_initialized():
            return 1

        return dist.get_world_size()

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        world_size = self.get_world_size()

        if world_size == 1:
            return tensor

        dist.all_reduce(tensor, op=op)
        return tensor

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.all_reduce(embed_onehot_sum)
            self.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))