import torch
from torch import nn

from utils import Encoder, Decoder, FixupResBlock
from utils import MyEncoder, MyDecoder, MyQuantize


class VQVAE(nn.Module):
    def __init__(self, params):
        super(VQVAE, self).__init__()
        self.p = params

        ## Self-Explanatory
        self.input_channels = 1

        ## Initial CNN Channels but its
        ## channels * 2**i for i in range n_blocks per bottleneck
        self.base_network_channels = 8

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


class MyVQVAE(nn.Module):
    def __init__(
        self,
        params,
        in_channel=1,
        channel=64,
        n_res_block=2,
        n_res_channel=64,
        embed_dim=32,
        n_embed=256,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = MyEncoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = MyEncoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv3d(channel, embed_dim, 1)
        self.quantize_t = MyQuantize(embed_dim, n_embed)
        self.dec_t = MyDecoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv3d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = MyQuantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose3d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = MyDecoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)
        diff = diff*0.25
        return dec, (diff, quant_t, quant_b)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        print(enc_b.shape, enc_t.shape)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 4, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 4, 1, 2, 3)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 4, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 4, 1, 2, 3)
        diff_b = diff_b.unsqueeze(0)

        print(quant_b.shape, quant_t.shape)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 4, 1, 2, 3)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 4, 1, 2, 3)

        dec = self.decode(quant_t, quant_b)

        return dec
