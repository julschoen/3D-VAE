import torch
from torch import nn

from utils import Encoder, Decoder, Quantizer, MyQuantize

class VQVAE(nn.Module):
    def __init__(
        self,
        params,
        decay=0.99,
    ):
        super().__init__()

        in_channel = 1
        channel = params.filter
        n_res_block = params.n_res_block
        n_res_channel = params.n_res_channel
        embed_dim = params.z_size
        n_embed = params.K


        self.enc_b = MyEncoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = MyEncoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv3d(channel, embed_dim, 1)
        self.quantize_t = Quantizer(n_embed, embed_dim)
        self.dec_t = MyDecoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv3d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantizer(n_embed, embed_dim)
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
        #diff = diff*0.25
        return dec, (diff, quant_t, quant_b)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t)
        diff_t, quant_t, id_t = self.quantize_t(quant_t)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b)
        diff_b, quant_b, id_b = self.quantize_b(quant_b)
        
        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        #quant_t = quant_t.permute(0, 4, 1, 2, 3)
        quant_b = self.quantize_b.embed_code(code_b)
        #quant_b = quant_b.permute(0, 4, 1, 2, 3)
        dec = self.decode(quant_t, quant_b)

        return dec
