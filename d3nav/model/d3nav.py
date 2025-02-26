from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.amp
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from .base_model import BaseModel

DEFAULT_DATATYPE = torch.float


class TrajectoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (B, 6, 2)
        self.mlp1 = nn.Linear(6 * 3, 6 * 3)
        self.mlp2 = nn.Linear(6 * 3, 6 * 3)
        self.layernorm = nn.LayerNorm(6 * 3)

        # Linear interpolation weights
        self.register_buffer("interp_weights", torch.linspace(0, 1, 256))

    def forward(self, x):
        # Input shape: (B, 6, 3)
        B, P, D = x.shape
        # Flatten the trajectory points
        x = x.reshape(B, P * D)

        # Apply MLPs with activations
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)
        x = self.layernorm(x)

        # Interpolate to 256 dimensions
        x = x.unsqueeze(1)  # (B, 18, 1)
        x = F.interpolate(x, size=256, mode="linear", align_corners=True)

        # Output shape: (B, 1, 256) - single token representation
        return x


class TrajectoryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (B, T, 256)
        self.mlp1 = nn.Linear(6 * 3, 6 * 3)
        self.mlp2 = nn.Linear(6 * 3, 6 * 3)
        self.layernorm = nn.LayerNorm(6 * 3)

    def forward(self, x):
        # Input shape: (B, T, 256)
        B, T, D = x.shape

        # If multiple tokens, average them
        if T > 1:
            x = torch.mean(x, dim=1)  # (B, 256)
        else:
            x = x.squeeze(1)  # (B, 256)

        # Interpolate back to 6*3 dimensions
        x = x.unsqueeze(1)  # (B, 1, 256)
        x = F.interpolate(x, size=18, mode="linear", align_corners=True)
        x = x.squeeze(1)  # (B, 18)

        # Apply MLPs with activations
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)
        x = self.layernorm(x)

        # Reshape to trajectory format
        x = x.reshape(B, 6, 3)

        x = x.cumsum(dim=1)

        return x


class ChunkedAttention(nn.Module):
    def __init__(self, chunk_size=43):
        super().__init__()
        self.chunk_size = chunk_size
        self.attention = nn.Linear(1025, 1)

    def forward(self, x):  # x: (B, 1032, 1025)
        # Reshape to chunks
        B = x.shape[0]
        x = x.reshape(B, -1, self.chunk_size, 1025)  # (B, 32, 32, 1025)

        # Attend within chunks
        weights = F.softmax(self.attention(x), dim=2)
        x = (x * weights).sum(dim=2)  # (B, 32, 1025)

        # Final attention across chunks
        weights = F.softmax(self.attention(x), dim=1)
        x = (x * weights).sum(dim=1)  # (B, 1025)
        return x.unsqueeze(1)


class D3Nav(BaseModel):
    """
    Building the d3nav Architecture from Comma AI's pretrained weights
    """

    def __init__(
        self,
        load_comma: bool = True,
        temporal_context: int = 8,
    ):
        super(D3Nav, self).__init__()

        # Load model GPT
        self.config_gpt = GPTConfig()
        model = GPT(self.config_gpt)
        if load_comma:
            model.load_state_dict_from_url(
                "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin",  # noqa
                assign=True,
            )
        self.model = model.to(dtype=DEFAULT_DATATYPE)

        # Load image decoder
        self.config_decoder = CompressorConfig()
        with torch.device("meta"):
            decoder = Decoder(self.config_decoder)
            if load_comma:
                decoder.load_state_dict_from_url(
                    "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin",  # noqa
                    assign=True,
                )
        self.decoder = decoder

        # Load image encoder
        self.config_encoder = CompressorConfig()

        with torch.device("meta"):
            encoder = Encoder(self.config_encoder)
            if load_comma:
                encoder.load_state_dict_from_url(
                    "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/encoder_pytorch_model.bin",  # noqa
                    assign=True,
                )

        self.encoder = encoder
        self.T: int = temporal_context

        self.dropout_rate: float = 0.0

        # Initialize trajectory encoder and decoder
        self.traj_encoder = TrajectoryEncoder()
        self.traj_decoder = TrajectoryDecoder()

        self.chunked_attention = ChunkedAttention(chunk_size=43)
        self.trajectory_proj = nn.Linear(1025, 256)

        # Freeze
        self.freeze_vqvae()
        self.freeze_gpt()
        self.freeze_traj_enc_dec()

    def unfreeze_last_n_layers(self, num_layers):
        """
        unfreeze last few lawers
        """
        # Get total number of layers
        total_layers = len(self.model.transformer.h)

        # Calculate which layers to apply LORA to (from the end)
        start_layer = total_layers - num_layers

        # Initialize for selected layers
        for i in range(start_layer, total_layers):
            layer = self.model.transformer.h[i]
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.model.lm_head.parameters():
            param.requires_grad = True

    def freeze_vqvae(self, requires_grad=False):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad
        for param in self.decoder.parameters():
            param.requires_grad = requires_grad

    def freeze_traj_enc_dec(self, requires_grad=False):
        """
        Freezes the trajectory encoder and decoder parameters
        """
        for param in self.traj_encoder.parameters():
            param.requires_grad = requires_grad
        for param in self.traj_decoder.parameters():
            param.requires_grad = requires_grad

    def freeze_traj_dec(self, requires_grad=False):
        """
        Freezes the trajectory encoder and decoder parameters
        """
        for param in self.traj_decoder.parameters():
            param.requires_grad = requires_grad

    def freeze_gpt(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def traj_quantize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Quantizes a trajectory by passing it through the encoder-decoder pair
        """
        encoded_traj = self.traj_encoder(trajectory)
        decoded_traj = self.traj_decoder(encoded_traj)
        return decoded_traj

    def quantize(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        z = self.encoder(x)
        xp = self.decoder(z)
        xp = xp.view(B, T, C, H, W)
        return xp

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes a video context as input
        Encodes it into token space
        Generates one token
        Decodes the trajectory
        """

        B, T, C, H, W = x.shape
        # assert T == self.T
        x = x.reshape(B * T, C, H, W)

        # z: torch.Tensor = self.encoder(x)
        z, z_history_feats = self.encoder(x, return_feats=True)
        z_history_feats = z_history_feats.reshape(B, T, 256, 8, 16)

        z = z.reshape(B, T, -1)

        # Create BOS tokens
        bos_tokens = torch.full(
            (B, T, 1),
            self.config_gpt.bos_token,
            dtype=z.dtype,
            device=z.device,
        )

        # Concatenate BOS tokens with z
        z = torch.cat([bos_tokens, z], dim=2)

        zp_l = []
        for index in range(B):
            zp_i = self.differentiable_generate(
                z[index].reshape(T * self.config_gpt.tokens_per_frame),
                self.config_gpt.tokens_per_frame,
            )

            zp_l.append(zp_i)
        zp: torch.Tensor = torch.stack(zp_l)
        zp = zp.reshape(B, 129 * self.T, self.config_gpt.dim + 1)

        # Chunked attention processing
        zp = self.chunked_attention(zp)  # (B, 1, 1025)

        # Project to expected dimension
        zp = self.trajectory_proj(zp.squeeze(1))  # (B, 256)

        planner_features = zp.reshape(B, 1, 256)

        # During training, we can use the actual trajectory
        decoded_traj = self.traj_decoder(planner_features)
        ego_trajectory = decoded_traj

        return ego_trajectory

    def differentiable_generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Single forward pass to get transformer
        features for trajectory decoder"""
        t = prompt.size(0)
        device = prompt.device

        prompt = prompt.view(self.T, 129)

        # prompt.shape: [1032]
        # self.T frames
        # Each frame is 129 tokens long

        # Single forward pass through transformer
        input_pos = torch.arange(0, t, device=device)
        transformer_output = self.model(
            prompt.view(1, -1),
            input_pos,
        )

        return transformer_output

    def forward_video(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes a video context as input
        Encodes it into token space
        Autoregresively generates the next frame
        Decodes the frame
        """

        B, T, C, H, W = x.shape
        # assert T == self.T
        x = x.reshape(B * T, C, H, W)

        # z: torch.Tensor = self.encoder(x)
        z, z_history_feats = self.encoder(x, return_feats=True)
        z_history_feats = z_history_feats.reshape(B, T, 256, 8, 16)

        z = z.to(dtype=torch.int32)
        z = z.reshape(B, T, -1)

        # Create BOS tokens
        bos_tokens = torch.full(
            (B, T, 1),
            self.config_gpt.bos_token,
            dtype=z.dtype,
            device=z.device,
        )

        # Concatenate BOS tokens with z
        z = torch.cat([bos_tokens, z], dim=2)
        zp_l = []
        for index in range(B):
            zp_i = self.model.generate(
                z[index].reshape(T * self.config_gpt.tokens_per_frame),
                self.config_gpt.tokens_per_frame,
            )
            zp_l.append(zp_i)
        zp: torch.Tensor = torch.cat(zp_l)
        zp = zp.reshape(B, self.config_gpt.tokens_per_frame)

        zp = zp[:, 1:]
        zp = zp.to(dtype=torch.int64)
        # TODO: straight through

        xp, z_feat = self.decoder(zp, return_feats=True)
        xp = xp.reshape(B, 1, C, H, W)

        return xp


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(
        dtype=torch.int
    )


def sample(logits):
    probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@dataclass
class GPTConfig:
    block_size: int = 20 * 129
    vocab_size: int = 1025
    n_layer: int = 24
    n_head: int = 16
    dim: int = 1024
    intermediate_size: int = 4 * 1024
    tokens_per_frame: int = 129

    @property
    def bos_token(self):
        return self.vocab_size - 1

    @property
    def head_dim(self):
        return self.dim // self.n_head


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size,
        max_seq_length,
        n_heads,
        head_dim,
        dtype=DEFAULT_DATATYPE,
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = Attention(config)
        self.mlp = FeedForward(config)
        self.ln_1 = nn.LayerNorm(config.dim)
        self.ln_2 = nn.LayerNorm(config.dim)

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attn(self.ln_1(x), mask, input_pos)
        out = h + self.mlp(self.ln_2(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.dim, 3 * config.dim, bias=True)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=True)
        self.kv_cache = None

    def forward(
        self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.c_attn(x).split(
            [self.config.dim, self.config.dim, self.config.dim], dim=-1
        )

        q = q.view(bsz, seqlen, self.config.n_head, self.config.head_dim)
        k = k.view(bsz, seqlen, self.config.n_head, self.config.head_dim)
        v = v.view(bsz, seqlen, self.config.n_head, self.config.head_dim)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
        else:
            mask = mask[:, :, :, :seqlen]

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.dim)
        return self.c_proj(y)


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(
            config.intermediate_size, config.dim, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig = GPTConfig()) -> None:
        super().__init__()
        self.config = config

        transformer = {
            "wte": nn.Embedding(config.vocab_size, config.dim),
            "wpe": nn.Embedding(config.block_size, config.dim),
            "h": nn.ModuleList(
                TransformerBlock(config) for _ in range(config.n_layer)
            ),
            "ln_f": nn.LayerNorm(config.dim),
        }

        self.transformer = nn.ModuleDict(transformer)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.block_size, config.block_size, dtype=torch.bool
                )
            ).view(1, 1, config.block_size, config.block_size),
        )

    def setup_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.transformer.h:
            b.attn.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_head,
                self.config.head_dim,
            )

        self.causal_mask = torch.tril(
            torch.ones(
                self.max_seq_length, self.max_seq_length, dtype=torch.bool
            )
        ).view(1, 1, self.max_seq_length, self.max_seq_length)

    def forward(
        self,
        idx: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if input_pos is None:
            input_pos = torch.arange(idx.shape[1], device=idx.device)

        mask = self.causal_mask[:, :, input_pos]
        wte_e = self.transformer.wte(idx)  # type: ignore
        wpe_e = self.transformer.wpe(input_pos)  # type: ignore
        x = wte_e + wpe_e

        for layer_index, layer in enumerate(
            self.transformer.h  # type: ignore
        ):
            x = layer(x, input_pos, mask)
        x = self.transformer.ln_f(x)  # type: ignore
        logits = self.lm_head(x)

        return logits

    def prefill(
        self, x: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        logits = self(x, input_pos)
        return sample(logits)[0]

    def decode_one_token(self, x: torch.Tensor, input_pos: torch.Tensor):
        assert input_pos.shape[-1] == 1
        logits = self(x, input_pos)
        return sample(logits)

    def decode_n_tokens(
        self,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
    ):
        new_tokens, new_probs = [], []
        for _ in range(num_new_tokens):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                next_token, next_prob = self.decode_one_token(
                    cur_token, input_pos
                )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)
        return new_tokens, new_probs

    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        t = prompt.size(0)
        T_new = t + max_new_tokens
        max_seq_length = self.config.block_size
        device, dtype = prompt.device, prompt.dtype
        with torch.device(device):
            self.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        seq = torch.empty(T_new, dtype=dtype, device=device).clone()
        seq[:t] = prompt
        input_pos = torch.arange(0, t, device=device)
        next_token = self.prefill(prompt.view(1, -1), input_pos).clone()
        seq[t] = next_token
        input_pos = torch.tensor([t], device=device, dtype=torch.int)
        generated_tokens, _ = self.decode_n_tokens(
            next_token.view(1, -1), input_pos, max_new_tokens - 1
        )
        seq[t + 1 :] = torch.cat(generated_tokens)
        return seq[t:]

    def load_state_dict_from_url(self, url, *args, **kwargs):
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            map_location="cpu",
            weights_only=True,
        )
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not any(
                [k.endswith(".attn.masked_bias"), k.endswith(".attn.bias")]
            )
        }
        state_dict["causal_mask"] = torch.tril(
            torch.ones(
                self.config.block_size,
                self.config.block_size,
                dtype=torch.bool,
            )
        ).view(1, 1, self.config.block_size, self.config.block_size)
        for k in state_dict.keys():
            if any(k.endswith(w) for w in transposed):
                state_dict[k] = torch.transpose(state_dict[k], 1, 0)
        self.load_state_dict(state_dict, *args, **kwargs)


# VQ-VAE


@dataclass
class CompressorConfig:
    in_channels: int = 3
    out_channels: int = 3
    ch_mult: Tuple[int, int, int, int, int] = (1, 1, 2, 2, 4)
    attn_resolutions: Tuple[int] = (16,)
    resolution: int = 256
    num_res_blocks: int = 2
    z_channels: int = 256
    vocab_size: int = 1024
    ch: int = 128
    dropout: float = 0.0

    @property
    def num_resolutions(self):
        return len(self.ch_mult)

    @property
    def quantized_resolution(self):
        return self.resolution // 2 ** (self.num_resolutions - 1)


def nonlinearity(x):  # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest"
        )
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw        w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(
            v, w_
        )  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim
        )
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )

    # the encode function
    def forward(self, inputs):
        b, s, c = inputs.shape
        flat_input = rearrange(inputs, "b s c -> (b s) c")

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embed(encoding_indices)
        quantized = rearrange(
            quantized, "(b s) c -> b s c", b=b, s=s, c=c
        ).contiguous()
        encoding_indices = rearrange(
            encoding_indices, "(b s) 1 -> b s", b=b, s=s
        )
        quantized = inputs + (quantized - inputs).detach()
        return quantized, encoding_indices

    # the decode function
    def decode(self, encoding_indices):
        b, s = encoding_indices.shape
        encoding_indices = rearrange(
            encoding_indices, "b s -> (b s) 1", b=b, s=s
        )
        quantized = self.embed(encoding_indices)
        quantized = rearrange(
            quantized, "(b s) c -> b s c", b=b, c=self._embedding_dim, s=s
        ).contiguous()
        encoding_indices = rearrange(
            encoding_indices, "(b s) 1 -> b s", b=b, s=s
        )
        return quantized, encoding_indices

    def embed(self, encoding_indices):
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=encoding_indices.device,
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight)
        return quantized


class Encoder(nn.Module):
    def __init__(self, config: CompressorConfig):
        super().__init__()
        self.config = config
        self.temb_ch = 0
        # downsampling
        self.conv_in = torch.nn.Conv2d(
            self.config.in_channels,
            self.config.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        curr_res = self.config.resolution
        in_ch_mult = (1,) + tuple(self.config.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.config.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.config.ch * in_ch_mult[i_level]
            block_out = self.config.ch * self.config.ch_mult[i_level]
            for _ in range(self.config.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=self.config.dropout,
                    )
                )
                block_in = block_out
                if curr_res in self.config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.config.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=self.config.dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=self.config.dropout,
        )
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            self.config.z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # quantizer
        self.quant_conv = torch.nn.Conv2d(
            self.config.z_channels, self.config.z_channels, 1
        )
        self.quantize = VectorQuantizer(
            self.config.vocab_size, self.config.z_channels
        )

    def forward(self, x, return_feats=False):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.config.num_resolutions):
            for i_block in range(self.config.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.config.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        # run the encoder part of VQ
        # h: (B, 256, 8, 16)
        h_feats = self.quant_conv(h)
        h = rearrange(h_feats, "b c h w -> b (h w) c")
        _, encoding_indices = self.quantize(h)
        if return_feats:
            return encoding_indices, h_feats
        return encoding_indices

    def load_state_dict_from_url(self, url, *args, **kwargs):
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", weights_only=True
        )
        self.load_state_dict(state_dict, *args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, config: CompressorConfig):
        super().__init__()
        self.temb_ch = 0
        self.config = config

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = (
            self.config.ch
            * self.config.ch_mult[self.config.num_resolutions - 1]
        )
        curr_res = self.config.quantized_resolution

        # quantizer
        self.post_quant_conv = torch.nn.Conv2d(
            config.z_channels, config.z_channels, 1
        )
        self.quantize = VectorQuantizer(config.vocab_size, config.z_channels)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            self.config.z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=self.config.dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=self.config.dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.config.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.config.ch * self.config.ch_mult[i_level]
            for _ in range(self.config.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=self.config.dropout,
                    )
                )
                block_in = block_out
                if curr_res in self.config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            self.config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, encoding_indices, return_feats=False):
        # run the decoder part of VQ
        z_feat, _ = self.quantize.decode(encoding_indices)
        z = rearrange(
            z_feat, "b (h w) c -> b c h w", w=self.config.quantized_resolution
        )
        z = self.post_quant_conv(z)
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.config.num_resolutions)):
            for i_block in range(self.config.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        img = ((h + 1.0) / 2.0) * 255.0

        if return_feats:
            return img, z_feat
        # scale
        return img

    def load_state_dict_from_url(self, url, *args, **kwargs):
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", weights_only=True
        )
        self.load_state_dict(state_dict, *args, **kwargs)


CROP_SIZE = (512, 256)
OUTPUT_SIZE = (CROP_SIZE[0] // 2, CROP_SIZE[1] // 2)
SCALE = 567 / 455
CY = 47.6


def write_video(frames_rgb, out, fps=20):
    size = frames_rgb[0].shape[:2][::-1]
    video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    for frame in frames_rgb:
        video.write(frame[..., ::-1])
    video.release()
    return out


def read_video(path):
    frames = []
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    video = np.stack(frames, axis=0)
    return video


@torch.no_grad()
def transpose_and_clip(tensors):
    tensors = np.array(tensors)
    tensors = np.transpose(tensors, (0, 2, 3, 1))
    tensors = np.clip(tensors, 0, 255).astype(np.uint8)
    return tensors


def transform_img(
    frame, output_size=OUTPUT_SIZE, crop_size=CROP_SIZE, scale=SCALE, cy=CY
):
    return cv2.resize(frame, output_size)


def d3nav_transform_img(frame: np.ndarray):
    """
    Accepts a CV2 image as input and transforms it to D3Nav input space
    Returns a numpy float matrix
    """
    frame = cv2.resize(frame, (512, 256))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform_img(frame)
    frame = np.transpose(frame, (2, 0, 1))
    frame = frame.astype(np.float32)
    return frame


def center_crop(frame: np.ndarray, crop_ratio: float):
    h, w, _ = frame.shape
    hs, ws = int(h * crop_ratio / 2), int(w * crop_ratio / 2)
    he, we = int(hs + h * (1 - crop_ratio)), int(ws + w * (1 - crop_ratio))

    he = h
    hs = int(2 * hs)

    frame = frame[hs:he, ws:we]

    return frame


@torch.no_grad()
def to_cv2_frame(img_torch: torch.Tensor) -> np.ndarray:
    img_np = img_torch.cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    # img_np = img_np + 128.0
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)

    return img_np


if __name__ == "__main__":
    import torch

    B, T, C, H, W = 1, 8, 3, 128, 256

    x = torch.zeros(
        (B, T, C, H, W), requires_grad=True
    )  # Add requires_grad=True
    model = D3Nav()
    model.unfreeze_last_n_layers(num_layers=3)

    print("x", x.shape, x.dtype, (x.min(), x.max()))

    traj = model(x)

    # Test gradient flow
    loss = traj.sum()
    loss.backward()

    print("x.grad is None:", x.grad is None)
    print("traj grad_fn:", traj.grad_fn)
