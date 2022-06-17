from itertools import chain
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

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

class FixupResBlock(nn.Module):
    # Adapted from:
    # https://github.com/hongyi-zhang/Fixup/blob/master/imagenet/models/fixup_resnet_imagenet.py#L20
    def __init__(self, in_channels, out_channels, mode, activation=nn.ELU, bottleneck_divisor=2):
        super().__init__()

        padding_mode = 'circular'

        assert mode in ("down", "same", "up", "out")
        self.mode = mode

        branch_channels = max(max(in_channels, out_channels) // bottleneck_divisor, 1)

        self.activation = activation()

        self.bias1a, self.bias1b, self.bias2a, self.bias2b, self.bias3a, self.bias3b, self.bias4 = (
            nn.Parameter(torch.zeros(1)) for _ in range(7)
        )
        self.scale = nn.Parameter(torch.ones(1))

        if mode == 'down':
            conv = nn.Conv3d
            kernel_size, stride, padding = 4, 2, 1
        elif mode in ('same', 'out'):
            conv = nn.Conv3d
            kernel_size, stride, padding = 3, 1, 1
        elif mode == 'up':
            conv = ResizeConv3D
            kernel_size, stride, padding = 3, 1, 1

        self.branch_conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.branch_conv2 = conv(
            in_channels=branch_channels,
            out_channels=branch_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            padding_mode=padding_mode
        )

        self.branch_conv3 = nn.Conv3d(
            in_channels=branch_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        if not (mode in ("same", "out") and in_channels == out_channels):
            self.bias1c, self.bias1d = (nn.Parameter(torch.zeros(1)) for _ in range(2))
            self.skip_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1 if mode != 'down' else 2),
                stride=(1 if mode != 'down' else 2),
                padding=0,
                bias=False
            )
        else:
            self.skip_conv = None


    def forward(self, input: torch.Tensor):

        out = self.activation(input + self.bias1a)
        out = self.branch_conv1(out + self.bias1b)

        out = self.activation(out + self.bias2a)
        out = self.branch_conv2(out + self.bias2b)

        out = self.activation(out + self.bias3a)
        out = self.branch_conv3(out + self.bias3b)

        out = out * self.scale + self.bias4

        out = out + (
            self.skip_conv(input + self.bias1c) + self.bias1d
            if self.skip_conv is not None
            else input
        )

        return out

    @torch.no_grad()
    def initialize_weights(self, num_layers):

        # branch_conv1
        weight = self.branch_conv1.weight
        nn.init.normal_(
            weight,
            mean=0,
            std=np.sqrt(2 / (weight.shape[0] * np.prod(weight.shape[2:]))) * num_layers ** (-0.5)
        )

        # branch_conv2
        nn.init.kaiming_normal_(self.branch_conv2.weight)

        # branch_conv3
        nn.init.constant_(self.branch_conv3.weight, val=0)
        # nn.init.kaiming_normal_(self.branch_conv3.weight)

        if self.skip_conv is not None:
            nn.init.xavier_normal_(self.skip_conv.weight)

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_down=2,
        resblock=FixupResBlock,
        n_post_downscale_blocks=0
    ):
        super().__init__()

        self.layers = nn.Sequential(*chain.from_iterable(
            (resblock(in_channels*2**i, in_channels*2**(i+1), mode='down'),
            *(resblock(in_channels*2**(i+1), in_channels*2**(i+1), mode='same')
              for _ in range(n_post_downscale_blocks)))
            for i in range(n_down)
        ))

    def forward(self, data):
        return self.layers(data)

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        aux_channels=0,
        n_up=2,
        mode='encoder',
        resblock=FixupResBlock,
        n_post_upscale_blocks=0,
    ):
        super().__init__()

        assert mode in ('encoder', 'decoder')

        # Some slight channel size shenanigans required for the first layer because of possible concat beforehand
        self.layers = nn.Sequential(*chain.from_iterable((
            (resblock(
                in_channels if i == n_up-1 else out_channels*(2**(i+1)),
                out_channels*(2**i),
                mode='up'),
            *(resblock(out_channels*(2**i), out_channels*(2**i), mode='same')
              for _ in range(n_post_upscale_blocks)))
            for i in range(n_up-1, -1, -1)
        )))

    def forward(self, data):
        return self.layers(data)

class PreQuantizationConditioning(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        has_aux,
        n_up=2,
        resblock=FixupResBlock,
        n_post_upscale_blocks=0,
    ):
        super().__init__()
        self.has_aux = has_aux#in_channels - out_channels * 8 != 0

        if self.has_aux:
            self.upsample = UpBlock(
                out_channels,
                out_channels,
                n_up=n_up,
                resblock=resblock,
                n_post_upscale_blocks=n_post_upscale_blocks,
            )
            self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.pre_q = resblock(in_channels, out_channels, mode='same')

    def forward(self, data, auxilary=None):
        assert self.has_aux is (auxilary is not None)

        if self.has_aux:
            data = self.proj(torch.cat([data, self.upsample(auxilary)], dim=1))

        return self.pre_q(data)

class Decoder(nn.Module):
    '''I'm sorry this logic is such a mess'''
    def __init__(
        self,
        out_channels,
        base_network_channels,
        n_enc=3,
        n_up_per_enc=2,
        n_post_q_blocks=0,
        n_post_upscale_blocks=0,
        resblock=FixupResBlock,
        embedding_dim=64
    ):
        super().__init__()

        self.up = nn.ModuleList()
        self.proj = nn.ModuleList()
        after_channels = base_network_channels

        for i in range(n_enc):
            before_channels = after_channels * 2 ** n_up_per_enc

            in_channels = embedding_dim + (before_channels if i != n_enc-1 else 0)

            if i != n_enc-1:
                self.proj.append(nn.Conv3d(in_channels, in_channels, kernel_size=1))

            self.up.append(nn.Sequential(
                *(resblock(in_channels, in_channels, mode='same')
                  for _ in range(n_post_q_blocks)),
                UpBlock(
                    in_channels=in_channels,
                    out_channels=after_channels,
                    n_up=n_up_per_enc,
                    mode='decoder',
                    resblock=resblock,
                    n_post_upscale_blocks=n_post_upscale_blocks
                ),
            ))

            after_channels = before_channels

        self.out = nn.Conv3d(base_network_channels, out_channels, kernel_size=1)

    def forward(self, quantizations):
        for i, (quantization, up) in enumerate(reversed(list(zip(quantizations, self.up)))):
            out = quantization if i == 0 else self.proj[-i](torch.cat([quantization, out], dim=1))
            out = up(out)

        out = self.out(out)

        return out

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        base_network_channels,
        num_embeddings,
        n_enc=3,
        n_down_per_enc=2,
        n_pre_q_blocks=0,
        n_post_upscale_blocks=0,
        n_post_downscale_blocks=0,
        resblock=FixupResBlock,
        embedding_dim=64
    ):
        super().__init__()

        self.parse_input = nn.Conv3d(in_channels, base_network_channels, kernel_size=1)

        before_channels = base_network_channels
        self.down, self.pre_quantize, self.pre_quantize_cond, self.quantize = (
            nn.ModuleList() for _ in range(4)
        )

        for i in range(n_enc):
            after_channels = before_channels * 2 ** n_down_per_enc

            self.down.append(
                DownBlock(
                    before_channels,
                    n_down_per_enc,
                    resblock=resblock,
                    n_post_downscale_blocks=n_post_downscale_blocks
                ),
            )

            self.pre_quantize_cond.append(
                PreQuantizationConditioning(
                    in_channels=after_channels + (embedding_dim if i != n_enc-1 else 0),
                    out_channels=embedding_dim,
                    has_aux= i != n_enc-1,
                    n_up=n_down_per_enc,
                    resblock=resblock,
                    n_post_upscale_blocks=n_post_upscale_blocks,
                )
            )
            self.pre_quantize.append(nn.Sequential(
                *(resblock(embedding_dim, embedding_dim, mode='same')
                  for _ in range(n_pre_q_blocks))
            ))
            self.quantize.append(
                Quantizer(num_embeddings=num_embeddings[i], embedding_dim=embedding_dim, commitment_cost=0.25)
            )

            before_channels = after_channels


    def forward(self, data):
        down = self.parse_input(data)
        downsampled = []
        for downblock in self.down:
            down = downblock(down)
            downsampled.append(down)


        aux = None
        quantizations = []
        for down, pre_quantize, pre_quantize_cond, quantize in reversed(list(zip(downsampled, self.pre_quantize, self.pre_quantize_cond, self.quantize))):
            quantizations.append(quantize(pre_quantize(pre_quantize_cond(down, aux))))

            _, aux, *_ = quantizations[-1]
        
        return reversed(quantizations)

class ResizeConv3D(nn.Conv3d):
    def __init__(self, *conv_args, **conv_kwargs):
        super(ResizeConv3D, self).__init__(*conv_args, **conv_kwargs)
        self.upsample = nn.Upsample(mode='trilinear', scale_factor=2, align_corners=False)

    def forward(self, input):
        return super().forward(self.upsample(input))

class Quantizer(nn.Module):
    # Code taken from:
    # https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I

    """
    EMA-updated Vector Quantizer
    """
    def __init__(
        self, num_embeddings : int, embedding_dim : int, commitment_cost : float, decay=0.99, laplace_alpha=1e-5
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

class MyEncoder(nn.Module):
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

class MyDecoder(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel
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

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

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