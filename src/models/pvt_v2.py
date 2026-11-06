'''
PyramidVisionTransformer V2 (PVTv2).

Ported and adapted from the HuggingFace Transformers implementation
(transformers.models.pvt_v2.modeling_pvt_v2), since `transformers` is not a
dependency of this project. Module names, forward logic and weight init are
kept faithful to that reference implementation.

Reference:
    "PVTv2: Improved Baselines with Pyramid Vision Transformer"
    https://arxiv.org/abs/2106.13797
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/pvt_v2/modeling_pvt_v2.py
'''

import json
import math
import os
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torchvision.ops import StochasticDepth

__all__ = [
    "PvtV2Config",
    "PvtV2Model",
    "PvtV2ForImageClassification",
    "pvt_v2_b0",
    "pvt_v2_b1",
    "pvt_v2_b2",
    "pvt_v2_b2_li",
    "pvt_v2_b3",
    "pvt_v2_b4",
    "pvt_v2_b5",
]


ACT2FN = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


@dataclass
class PvtV2Config:
    num_channels: int = 3
    num_encoder_blocks: int = 4
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    sr_ratios: List[int] = field(default_factory=lambda: [8, 4, 2, 1])
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 64, 160, 256])
    patch_sizes: List[int] = field(default_factory=lambda: [7, 3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [4, 2, 2, 2])
    num_attention_heads: List[int] = field(default_factory=lambda: [1, 2, 5, 8])
    mlp_ratios: List[int] = field(default_factory=lambda: [8, 8, 4, 4])
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    drop_path_rate: float = 0.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    linear_attention: bool = False
    num_labels: int = 1000


@dataclass
class PvtV2ModelOutput:
    last_hidden_state: Tensor
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None


class PvtV2OverlapPatchEmbeddings(nn.Module):
    """Image to overlapping patch embedding, used at the start of every stage."""

    def __init__(self, config: PvtV2Config, layer_idx: int):
        super().__init__()
        patch_size = config.patch_sizes[layer_idx]
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        stride = config.strides[layer_idx]
        num_channels = (
            config.num_channels if layer_idx == 0 else config.hidden_sizes[layer_idx - 1]
        )
        hidden_size = config.hidden_sizes[layer_idx]

        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: Tensor):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class PvtV2DepthWiseConv(nn.Module):
    """Depth-wise convolution used to inject positional information into the Mix-FFN."""

    def __init__(self, config: PvtV2Config, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, hidden_states: Tensor, height: int, width: int) -> Tensor:
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class PvtV2SelfAttention(nn.Module):
    """Spatial-reduction (or linear, when config.linear_attention) self-attention."""

    def __init__(
        self,
        config: PvtV2Config,
        hidden_size: int,
        num_attention_heads: int,
        spatial_reduction_ratio: int,
    ):
        super().__init__()
        self.linear_attention = config.linear_attention
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) is not a multiple of num_attention_heads ({self.num_attention_heads})"
            )
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)

        self.spatial_reduction_ratio = spatial_reduction_ratio
        if self.linear_attention:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.spatial_reduction = nn.Conv2d(
                self.hidden_size, self.hidden_size, kernel_size=1, stride=1
            )
            self.layer_norm = nn.LayerNorm(self.hidden_size)
            self.act = nn.GELU()
        elif spatial_reduction_ratio > 1:
            self.spatial_reduction = nn.Conv2d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=spatial_reduction_ratio,
                stride=spatial_reduction_ratio,
            )
            self.layer_norm = nn.LayerNorm(self.hidden_size)

    def transpose_for_scores(self, hidden_states: Tensor) -> Tensor:
        new_shape = hidden_states.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: Tensor, height: int, width: int, output_attentions: bool = False
    ):
        batch_size, seq_len, num_channels = hidden_states.shape
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.linear_attention:
            reduced = hidden_states.permute(0, 2, 1).reshape(
                batch_size, num_channels, height, width
            )
            reduced = (
                self.spatial_reduction(self.pool(reduced))
                .reshape(batch_size, num_channels, -1)
                .permute(0, 2, 1)
            )
            reduced = self.act(self.layer_norm(reduced))
        elif self.spatial_reduction_ratio > 1:
            reduced = hidden_states.permute(0, 2, 1).reshape(
                batch_size, num_channels, height, width
            )
            reduced = (
                self.spatial_reduction(reduced)
                .reshape(batch_size, num_channels, -1)
                .permute(0, 2, 1)
            )
            reduced = self.layer_norm(reduced)
        else:
            reduced = hidden_states

        key_layer = self.transpose_for_scores(self.key(reduced))
        value_layer = self.transpose_for_scores(self.value(reduced))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).reshape(batch_size, seq_len, num_channels)
        context_layer = self.proj(context_layer)
        context_layer = self.proj_drop(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class PvtV2ConvFeedForwardNetwork(nn.Module):
    """Mix-FFN: MLP with a depth-wise conv sandwiched in between the two linear layers."""

    def __init__(
        self,
        config: PvtV2Config,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features

        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = PvtV2DepthWiseConv(config, hidden_features)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]()
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # The linear-attention variant additionally applies a ReLU right after dense1.
        self.relu = nn.ReLU() if config.linear_attention else nn.Identity()

    def forward(self, hidden_states: Tensor, height: int, width: int) -> Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PvtV2BlockLayer(nn.Module):
    """Pre-norm transformer block: SR-Attention + Mix-FFN, each with a residual connection."""

    def __init__(self, config: PvtV2Config, layer_idx: int, drop_path: float = 0.0):
        super().__init__()
        hidden_size = config.hidden_sizes[layer_idx]
        num_attention_heads = config.num_attention_heads[layer_idx]
        spatial_reduction_ratio = config.sr_ratios[layer_idx]
        mlp_ratio = config.mlp_ratios[layer_idx]

        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = PvtV2SelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            spatial_reduction_ratio=spatial_reduction_ratio,
        )
        self.drop_path = (
            StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = PvtV2ConvFeedForwardNetwork(
            config=config, in_features=hidden_size, hidden_features=mlp_hidden_size
        )

    def forward(
        self, hidden_states: Tensor, height: int, width: int, output_attentions: bool = False
    ):
        self_attention_outputs = self.attention(
            hidden_states=self.layer_norm_1(hidden_states),
            height=height,
            width=width,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = hidden_states + self.drop_path(attention_output)

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)
        layer_output = hidden_states + self.drop_path(mlp_output)

        return (layer_output,) + outputs


class PvtV2EncoderLayer(nn.Module):
    """A single PVTv2 stage: overlapping patch embedding + N transformer blocks + final layer norm."""

    def __init__(self, config: PvtV2Config, layer_idx: int):
        super().__init__()
        self.patch_embedding = PvtV2OverlapPatchEmbeddings(config=config, layer_idx=layer_idx)

        # Stochastic depth decay rule, computed over the full network so that drop
        # probability increases linearly across all blocks of all stages.
        drop_path_decays = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()
        block_start = sum(config.depths[:layer_idx])
        self.blocks = nn.ModuleList(
            [
                PvtV2BlockLayer(
                    config=config, layer_idx=layer_idx, drop_path=drop_path_decays[block_start + i]
                )
                for i in range(config.depths[layer_idx])
            ]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_sizes[layer_idx])

    def forward(self, hidden_states: Tensor, output_attentions: bool = False):
        all_self_attentions = () if output_attentions else None
        hidden_states, height, width = self.patch_embedding(hidden_states)
        for block in self.blocks:
            layer_outputs = block(hidden_states, height, width, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states, height, width, all_self_attentions


class PvtV2Encoder(nn.Module):
    """Stacks the 4 PVTv2 stages, feeding the (B, C, H, W) output of stage i into stage i+1."""

    def __init__(self, config: PvtV2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [PvtV2EncoderLayer(config, i) for i in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_hidden_states = [None] if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]
        hidden_states = pixel_values
        for layer in self.layers:
            hidden_states, height, width, attentions = layer(hidden_states, output_attentions)
            if output_attentions:
                all_self_attentions += (attentions,)
            # Reshape the sequence of patch tokens back into a spatial feature map so it
            # can feed the next stage's (conv-based) overlapping patch embedding.
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        return PvtV2ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


def _init_weights(module: nn.Module, initializer_range: float = 0.02):
    if isinstance(module, nn.Linear):
        module.weight.data = nn.init.trunc_normal_(
            module.weight.data, mean=0.0, std=initializer_range
        )
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()


class PvtV2Model(nn.Module):
    """PVTv2 hierarchical backbone. Returns a (B, C, H, W) feature map per stage."""

    def __init__(self, config: PvtV2Config):
        super().__init__()
        self.config = config
        self.encoder = PvtV2Encoder(config)
        self.apply(lambda m: _init_weights(m, config.initializer_range))

    def forward(
        self,
        pixel_values: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        return self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class PvtV2ForImageClassification(nn.Module):
    """PVTv2 backbone with a linear classification head on top of the pooled last stage."""

    def __init__(self, config: PvtV2Config):
        super().__init__()
        self.config = config
        self.pvt_v2 = PvtV2Model(config)
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )
        self.apply(lambda m: _init_weights(m, config.initializer_range))

    def forward(self, pixel_values: Tensor) -> Tensor:
        outputs = self.pvt_v2(pixel_values)
        sequence_output = outputs.last_hidden_state  # (B, C, H, W)

        batch_size, num_channels = sequence_output.shape[:2]
        sequence_output = sequence_output.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
        pooled_output = sequence_output.mean(dim=1)

        logits = self.classifier(pooled_output)
        return logits


# ---------------------------------------------------------------------------
# Model variants, matching the official PVTv2 / HuggingFace configurations.
# ---------------------------------------------------------------------------

PVT_V2_CONFIGS = {
    "b0": dict(hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2], mlp_ratios=[8, 8, 4, 4]),
    "b1": dict(hidden_sizes=[64, 128, 320, 512], depths=[2, 2, 2, 2], mlp_ratios=[8, 8, 4, 4]),
    "b2": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 6, 3], mlp_ratios=[8, 8, 4, 4]),
    "b2_li": dict(
        hidden_sizes=[64, 128, 320, 512],
        depths=[3, 4, 6, 3],
        mlp_ratios=[8, 8, 4, 4],
        linear_attention=True,
    ),
    "b3": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 18, 3], mlp_ratios=[8, 8, 4, 4]),
    "b4": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 8, 27, 3], mlp_ratios=[8, 8, 4, 4]),
    "b5": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 6, 40, 3], mlp_ratios=[4, 4, 4, 4]),
}


# ---------------------------------------------------------------------------
# Loading official pretrained weights from the HuggingFace Hub
# (https://huggingface.co/OpenGVLab), without depending on `transformers` or
# `safetensors` -- only `requests` (already a project dependency) is used.
# ---------------------------------------------------------------------------

_HF_REPO_IDS = {
    "b0": "OpenGVLab/pvt_v2_b0",
    "b1": "OpenGVLab/pvt_v2_b1",
    "b2": "OpenGVLab/pvt_v2_b2",
    "b2_li": "OpenGVLab/pvt_v2_b2_li",
    "b3": "OpenGVLab/pvt_v2_b3",
    "b4": "OpenGVLab/pvt_v2_b4",
    "b5": "OpenGVLab/pvt_v2_b5",
}

_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _load_safetensors(path: str) -> dict:
    """
    Minimal pure-Python `.safetensors` reader: an 8-byte little-endian header
    length, a JSON header describing each tensor's dtype/shape/byte offsets,
    followed by the raw tensor bytes.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()

    state_dict = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = _SAFETENSORS_DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        tensor = torch.frombuffer(bytearray(data[start:end]), dtype=dtype).reshape(info["shape"])
        state_dict[name] = tensor
    return state_dict


def _download_hf_file(
    repo_id: str, filename: str, cache_dir: Optional[str] = None, progress: bool = True
) -> str:
    import requests

    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "pvt_v2")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{repo_id.replace('/', '__')}__{filename}")
    if not os.path.exists(local_path):
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", 0))
        chunks = response.iter_content(chunk_size=1 << 20)
        if progress:
            from tqdm import tqdm

            chunks = tqdm(
                chunks,
                total=(total_bytes // (1 << 20)) + 1,
                unit="MB",
                desc=f"Downloading {filename}",
            )

        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)
        os.replace(tmp_path, local_path)
    return local_path


def _load_pretrained_weights(
    model: nn.Module, variant: str, num_labels: int, progress: bool, cache_dir: Optional[str]
) -> None:
    """Download the official checkpoint and load it in-place into `model`."""
    if variant not in _HF_REPO_IDS:
        raise ValueError(f"No pretrained checkpoint is available for PVTv2 variant '{variant}'.")

    repo_id = _HF_REPO_IDS[variant]
    weights_path = _download_hf_file(
        repo_id, "model.safetensors", cache_dir=cache_dir, progress=progress
    )
    raw_state_dict = _load_safetensors(weights_path)

    # Checkpoint keys look like "pvt_v2.encoder.layers.0...." and "classifier....".
    encoder_state_dict = {
        k[len("pvt_v2.") :]: v for k, v in raw_state_dict.items() if k.startswith("pvt_v2.")
    }
    classifier_state_dict = {
        k[len("classifier.") :]: v
        for k, v in raw_state_dict.items()
        if k.startswith("classifier.")
    }

    backbone = model.pvt_v2 if isinstance(model, PvtV2ForImageClassification) else model
    backbone.load_state_dict(encoder_state_dict, strict=True)

    if isinstance(model, PvtV2ForImageClassification):
        if num_labels == 1000 and classifier_state_dict:
            model.classifier.load_state_dict(classifier_state_dict, strict=True)
        else:
            print(
                f"[pvt_v2] num_labels={num_labels} does not match the checkpoint's 1000-way head; "
                f"classifier left randomly initialized."
            )


def _build_pvt_v2(
    variant: str,
    pretrained: bool = False,
    progress: bool = True,
    num_labels: int = 0,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    config = PvtV2Config(num_labels=num_labels, **PVT_V2_CONFIGS[variant], **kwargs)
    model = PvtV2ForImageClassification(config) if num_labels > 0 else PvtV2Model(config)

    if pretrained:
        _load_pretrained_weights(
            model, variant, num_labels=num_labels, progress=progress, cache_dir=cache_dir
        )

    return model


def pvt_v2_b0(pretrained: bool = False, progress: bool = True, **kwargs) -> nn.Module:
    return _build_pvt_v2("b0", pretrained=pretrained, progress=progress, **kwargs)


def pvt_v2_b1(pretrained: bool = False, progress: bool = True, **kwargs) -> nn.Module:
    return _build_pvt_v2("b1", pretrained=pretrained, progress=progress, **kwargs)


def pvt_v2_b2(pretrained: bool = False, progress: bool = True, **kwargs) -> nn.Module:
    return _build_pvt_v2("b2", pretrained=pretrained, progress=progress, **kwargs)


def pvt_v2_b2_li(pretrained: bool = False, progress: bool = True, **kwargs) -> nn.Module:
    """Linear-complexity variant: pools K/V to a fixed 7x7 grid instead of strided conv SR."""
    return _build_pvt_v2("b2_li", pretrained=pretrained, progress=progress, **kwargs)


def pvt_v2_b3(pretrained: bool = False, progress: bool = True, **kwargs) -> nn.Module:
    return _build_pvt_v2("b3", pretrained=pretrained, progress=progress, **kwargs)


def pvt_v2_b4(pretrained: bool = False, progress: bool = True, **kwargs) -> nn.Module:
    return _build_pvt_v2("b4", pretrained=pretrained, progress=progress, **kwargs)


def pvt_v2_b5(pretrained: bool = False, progress: bool = True, **kwargs) -> nn.Module:
    return _build_pvt_v2("b5", pretrained=pretrained, progress=progress, **kwargs)


if __name__ == "__main__":
    model = pvt_v2_b2(pretrained=True)
    dummy = torch.randn(2, 3, 512, 640)
    outputs = model(dummy, output_hidden_states=True)
    print("Last hidden state shape:", outputs.last_hidden_state.shape)
    print("Multi-scale feature shapes:", [h.shape for h in outputs.hidden_states if h is not None])

    # clf_model = pvt_v2_b0(pretrained=True, num_labels=1000)
    # logits = clf_model(dummy)
    # print("Classification logits shape:", logits.shape)
