from functools import partial
from typing import Tuple
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from vqvae.layers import Encoder, Decoder, FixupResBlock, PreActFixupResBlock, EvonormResBlock, ExtractCenterCylinder

from utils.argparse_helpers import booltype



class VQVAE(nn.Module):
    def __init__(self, args: Namespace):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(
            in_channels=self.input_channels,
            base_network_channels=self.base_network_channels,
            n_enc=self.n_bottleneck_blocks,
            n_down_per_enc=self.n_blocks_per_bottleneck,
            n_pre_q_blocks=self.n_pre_quantization_blocks,
            n_post_downscale_blocks=self.n_post_downscale_blocks,
            n_post_upscale_blocks=self.n_post_upscale_blocks,
            num_embeddings=self.num_embeddings,
            resblock=self.resblock,

        )
        self.decoder = Decoder(
            out_channels=self.output_channels,
            base_network_channels=self.base_network_channels,
            n_enc=self.n_bottleneck_blocks,
            n_up_per_enc=self.n_blocks_per_bottleneck,
            n_post_q_blocks=self.n_post_quantization_blocks,
            n_post_upscale_blocks=self.n_post_upscale_blocks,
            resblock=self.resblock,
        )


        def init_fixupresblock(layer):
            if isinstance(layer, FixupResBlock) or isinstance(layer, PreActFixupResBlock):
                layer.initialize_weights(num_layers=self.num_layers)
        self.apply(init_fixupresblock)

    def forward(self, data):
        commitment_loss, quantizations, encoding_idx = zip(*self.encode(data))

        decoded = self.decode(quantizations)
        return decoded, (commitment_loss, quantizations, encoding_idx)

    def encode(self, data):
        return self.encoder(data)

    def decode(self, quantizations):
        return self.decoder(quantizations)


    def _parse_input_args(self, args: Namespace):
        assert args.metric in self.supported_metrics

    

        self.input_channels = args.input_channels
        self.output_channels = args.input_channels
        self.base_network_channels = args.base_network_channels
        self.n_bottleneck_blocks = args.n_bottleneck_blocks
        self.n_blocks_per_bottleneck = args.n_downscales_per_bottleneck
        self.n_pre_quantization_blocks = args.n_pre_quantization_blocks
        self.n_post_quantization_blocks = args.n_post_quantization_blocks
        self.n_post_upscale_blocks = args.n_post_upscale_blocks
        self.n_post_downscale_blocks = args.n_post_downscale_blocks

        assert len(args.num_embeddings) in (1, args.n_bottleneck_blocks)
        if len(args.num_embeddings) == 1:
            self.num_embeddings = [args.num_embeddings[0] for _ in range(args.n_bottleneck_blocks)]
        else:
            self.num_embeddings = args.num_embeddings

        resblocks = {'regular': FixupResBlock, 'pre-activation': PreActFixupResBlock, 'evonorm': EvonormResBlock}
        self.resblock = resblocks['pre-activation']

        # num_layers is defined as the longest path through the model
        n_down = args.n_bottleneck_blocks * args.n_downscales_per_bottleneck
        self.num_layers = (
            2 # input + output layer
            + 2 * n_down # down and up
            + args.n_pre_quantization_blocks
            + args.n_post_quantization_blocks
            + args.n_post_downscale_blocks * n_down
            + args.n_post_upscale_blocks * n_down
            + 1 # pre-activation block
        )

        self.pre_loss_f = ExtractCenterCylinder() if args.extract_center_cylinder else None
