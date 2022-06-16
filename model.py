import torch
from torch import nn

from utils import Encoder, Decoder, PreActFixupResBlock, FixupResBlock




class VQVAE(nn.Module):
    def __init__(self, params):
        super(VQVAE, self).__init__()
        self.p = params
        self.input_channels = 1
        self.base_network_channels = 4
        self.n_bottleneck_blocks = 2
        self.n_blocks_per_bottleneck = 2
        self.n_pre_quantization_blocks = 0
        self.n_post_downscale_blocks = 0
        self.n_post_upscale_blocks = 0
        self.num_embeddings = [self.p.z_size for _ in range(self.n_bottleneck_blocks)]
        if self.p.res == 0:
            self.resblock = PreActFixupResBlock
        else:
            self.resblock = FixupResBlock
        self.n_post_quantization_blocks = 0
        # num_layers is defined as the longest path through the model
        n_down = self.n_bottleneck_blocks * self.n_blocks_per_bottleneck
        self.num_layers = (
            2 # input + output layer
            + 2 * n_down # down and up
            + self.n_pre_quantization_blocks
            + self.n_post_quantization_blocks
            + self.n_post_downscale_blocks * n_down
            + self.n_post_upscale_blocks * n_down
            + 1 # pre-activation block
        )

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
            out_channels=self.input_channels,
            base_network_channels=self.base_network_channels,
            n_enc=self.n_bottleneck_blocks,
            n_up_per_enc=self.n_blocks_per_bottleneck,
            n_post_q_blocks=self.n_post_quantization_blocks,
            n_post_upscale_blocks=self.n_post_upscale_blocks,
            resblock=self.resblock,
        )


        def init_fixupresblock(layer):
            if isinstance(layer, PreActFixupResBlock): #isinstance(layer, FixupResBlock) or 
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
