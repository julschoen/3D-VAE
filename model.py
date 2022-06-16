import torch
from torch import nn

from utils import Encoder, Decoder, FixupResBlock


class VQVAE(nn.Module):
    def __init__(self, params):
        super(VQVAE, self).__init__()
        self.p = params

        ## Self-Explanatory
        self.input_channels = 1

        ## Initial CNN Channels but its
        ## channels * 2**i for i in range n_blocks per bottleneck
        self.base_network_channels = 4

        ## How Many Latent Spaces
        self.n_bottleneck_blocks = 2

        ## How many blocks between each latent space
        self.n_blocks_per_bottleneck = 2

        ## Num blocks after downscale before quantization
        self.n_pre_quantization_blocks = 0

        ## How many blocks before upscale after quantization in Decoder
        self.n_post_quantization_blocks = 0

        ## How many blocks not downscale in Encoder Block
        self.n_post_downscale_blocks = 0

        ## How many blocks not upscale in Decoder Block
        self.n_post_upscale_blocks = 0

        ## This is codebook size referred to as K in Paper
        self.num_embeddings = [self.p.z_size for i in range(self.n_bottleneck_blocks)]

        ## How large are words in codebook
        self.code_dim=64

        
        ## Just used for layer weight init (longest path throught net)
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
            embedding_dim=self.code_dim

        )
        self.decoder = Decoder(
            out_channels=self.input_channels,
            base_network_channels=self.base_network_channels,
            n_enc=self.n_bottleneck_blocks,
            n_up_per_enc=self.n_blocks_per_bottleneck,
            n_post_q_blocks=self.n_post_quantization_blocks,
            n_post_upscale_blocks=self.n_post_upscale_blocks,
            embedding_dim=self.code_dim
        )


        def init_fixupresblock(layer):
            if isinstance(layer, FixupResBlock):
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
