import math
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import RSTB, MultistageMaskedConv2d
from timm.models.layers import trunc_normal_

from .utils import conv, deconv, update_registered_buffers, Demultiplexer, Multiplexer

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class TinyLIC(nn.Module):
    """Lossy image compression framework from Ming Lu and Zhan Ma
    "High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation"

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=128, M=192):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(128,128),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(64,64),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(32,32),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
        )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(16,16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(8,8),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
        )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(4,4),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(4,4),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(8,8),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)
        
        self.g_s0 = RSTB(dim=M,
                        input_resolution=(16,16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = RSTB(dim=N,
                        input_resolution=(32,32),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = RSTB(dim=N,
                        input_resolution=(64,64),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
        )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = RSTB(dim=N,
                        input_resolution=(128,128),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.context_prediction_1 = MultistageMaskedConv2d(
                M, M*2, kernel_size=3, padding=1, stride=1, mask_type='A'
        )
        self.context_prediction_2 = MultistageMaskedConv2d(
                M, M*2, kernel_size=3, padding=1, stride=1, mask_type='B'
        )
        self.context_prediction_3 = MultistageMaskedConv2d(
                M, M*2, kernel_size=3, padding=1, stride=1, mask_type='C'
        )

        self.entropy_parameters = nn.Sequential(
                conv(M*24//3, M*18//3, 1, 1),
                nn.GELU(),
                conv(M*18//3, M*12//3, 1, 1),
                nn.GELU(),
                conv(M*12//3, M*6//3, 1, 1),
        ) 

        self.apply(self._init_weights)   

    def g_a(self, x, x_size=None):
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)
        x = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x = self.g_a2(x)
        x = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        x = self.g_a4(x)
        x = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        x = self.g_a6(x)
        x = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        return x

    def g_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        x = self.g_s1(x)
        x = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        x = self.g_s3(x)
        x = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        x = self.g_s5(x)
        x = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        x = self.g_s7(x)
        return x

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size)
        z = self.h_a(y, x_size)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat, x_size)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        y_1 = y_hat.clone()
        y_1[:, :, 0::2, 1::2] = 0
        y_1[:, :, 1::2, :] = 0
        ctx_params_1 = self.context_prediction_1(y_1)
        ctx_params_1[:, :, 0::2, :] = 0
        ctx_params_1[:, :, 1::2, 0::2] = 0

        y_2 = y_hat.clone()
        y_2[:, :, 0::2, 1::2] = 0
        y_2[:, :, 1::2, 0::2] = 0
        ctx_params_2 = self.context_prediction_2(y_2)
        ctx_params_2[:, :, 0::2, 0::2] = 0
        ctx_params_2[:, :, 1::2, :] = 0

        y_3 = y_hat.clone()
        y_3[:, :, 1::2, 0::2] = 0
        ctx_params_3 = self.context_prediction_3(y_3)
        ctx_params_3[:, :, 0::2, :] = 0
        ctx_params_3[:, :, 1::2, 1::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat = self.g_s(y_hat, x_size)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat, x_size)

        zero_ctx_params = torch.zeros_like(params).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params, zero_ctx_params, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)
        
        y_1 = y_hat.clone()
        y_1[:, :, 0::2, 1::2] = 0
        y_1[:, :, 1::2, :] = 0
        ctx_params_1 = self.context_prediction_1(y_1)
        ctx_params_1[:, :, 0::2, :] = 0
        ctx_params_1[:, :, 1::2, 0::2] = 0
        
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, zero_ctx_params, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)

        y_2 = y_hat.clone()
        y_2[:, :, 0::2, 1::2] = 0
        y_2[:, :, 1::2, 0::2] = 0
        ctx_params_2 = self.context_prediction_2(y_2)
        ctx_params_2[:, :, 0::2, 0::2] = 0
        ctx_params_2[:, :, 1::2, :] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)

        y_3 = y_hat.clone()
        y_3[:, :, 1::2, 0::2] = 0
        ctx_params_3 = self.context_prediction_3(y_3)
        ctx_params_3[:, :, 0::2, :] = 0
        ctx_params_3[:, :, 1::2, 1::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y1, y2, y3, y4 = Demultiplexer(y)
        scales_hat_y1, scales_hat_y2, scales_hat_y3, scales_hat_y4 = Demultiplexer(scales_hat)
        means_hat_y1, means_hat_y2, means_hat_y3, means_hat_y4 = Demultiplexer(means_hat)

        indexes_y1 = self.gaussian_conditional.build_indexes(scales_hat_y1)
        indexes_y2 = self.gaussian_conditional.build_indexes(scales_hat_y2)
        indexes_y3 = self.gaussian_conditional.build_indexes(scales_hat_y3)
        indexes_y4 = self.gaussian_conditional.build_indexes(scales_hat_y4)

        y1_strings = self.gaussian_conditional.compress(y1, indexes_y1, means=means_hat_y1)
        y2_strings = self.gaussian_conditional.compress(y2, indexes_y2, means=means_hat_y2)
        y3_strings = self.gaussian_conditional.compress(y3, indexes_y3, means=means_hat_y3)
        y4_strings = self.gaussian_conditional.compress(y4, indexes_y4, means=means_hat_y4)

        return {
            "strings": [y1_strings, y2_strings, y3_strings, y4_strings, z_strings],
            "shape": z.size()[-2:],
        }
    
    def decompress(self, strings, shape):
        """
        See Figure 5. Illustration of the proposed two-pass decoding.
        """
        assert isinstance(strings, list) and len(strings) == 5
        z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
        params = self.h_s(z_hat, (z_hat.shape[2]*64, z_hat.shape[3]*64))

        # Stage 0:
        zero_ctx_params = torch.zeros_like(params).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params, zero_ctx_params, zero_ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat_y1, _, _, _ = Demultiplexer(scales_hat)
        means_hat_y1, _, _, _ = Demultiplexer(means_hat)

        indexes_y1 = self.gaussian_conditional.build_indexes(scales_hat_y1)
        _y1 = self.gaussian_conditional.decompress(strings[0], indexes_y1, means=means_hat_y1)     # [1, 384, 8, 8]
        y1 = Multiplexer(_y1, torch.zeros_like(_y1), torch.zeros_like(_y1), torch.zeros_like(_y1))    # [1, 192, 16, 16]
        
        # Stage 1:
        ctx_params_1 = self.context_prediction_1(y1)
        ctx_params_1[:, :, 0::2, :] = 0
        ctx_params_1[:, :, 1::2, 0::2] = 0
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, zero_ctx_params, zero_ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, scales_hat_y2, _, _ = Demultiplexer(scales_hat)
        _, means_hat_y2, _, _ = Demultiplexer(means_hat)

        indexes_y2 = self.gaussian_conditional.build_indexes(scales_hat_y2)
        _y2 = self.gaussian_conditional.decompress(strings[1], indexes_y2, means=means_hat_y2)     # [1, 384, 8, 8]
        y2 = Multiplexer(torch.zeros_like(_y2), _y2, torch.zeros_like(_y2), torch.zeros_like(_y2))    # [1, 192, 16, 16]

        # Stage 2:
        ctx_params_2 = self.context_prediction_2(y1 + y2)
        ctx_params_2[:, :, 0::2, 0::2] = 0
        ctx_params_2[:, :, 1::2, :] = 0
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, zero_ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, _, scales_hat_y3, _ = Demultiplexer(scales_hat)
        _, _, means_hat_y3, _ = Demultiplexer(means_hat)

        indexes_y3 = self.gaussian_conditional.build_indexes(scales_hat_y3)
        _y3 = self.gaussian_conditional.decompress(strings[2], indexes_y3, means=means_hat_y3)     # [1, 384, 8, 8]
        y3 = Multiplexer(torch.zeros_like(_y3), torch.zeros_like(_y3), _y3, torch.zeros_like(_y3))    # [1, 192, 16, 16]

        # Stage 3:
        ctx_params_3 = self.context_prediction_3(y1 + y2 + y3)
        ctx_params_3[:, :, 0::2, :] = 0
        ctx_params_3[:, :, 1::2, 1::2] = 0
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, _, _, scales_hat_y4 = Demultiplexer(scales_hat)
        _, _, _, means_hat_y4 = Demultiplexer(means_hat)

        indexes_y4 = self.gaussian_conditional.build_indexes(scales_hat_y4)
        _y4 = self.gaussian_conditional.decompress(strings[3], indexes_y4, means=means_hat_y4)     # [1, 384, 8, 8]
        y4 = Multiplexer(torch.zeros_like(_y4), torch.zeros_like(_y4), torch.zeros_like(_y4), _y4)    # [1, 192, 16, 16]
        
        # gather
        y_hat = y1 + y2 + y3 + y4
        x_hat = self.g_s(y_hat, (y_hat.shape[2]*16, y_hat.shape[3]*16)).clamp_(0, 1)

        return {
            "x_hat": x_hat,
        }