import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from rotary_embedding_torch import RotaryEmbedding


@dataclass
class RoutingConfig:
    image_size: int = 1024
    n_layers: int = 12
    embedd_dim: int = 256
    n_heads: int = 8
    factor: int = 4
    channels: int = 3
    top_k: int = 128
    experts: int = 4
    bias: bool = False
    classes: int = 2
    cnn_dim: int = 96
    max_pool_size: int = 8
    max_pool_stride: int = 8
    num_patchs: int = 1024
    dropout: float = 0.1


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.999)


def pair(x):
    if isinstance(x, tuple):
        return x
    else:
        return (x, x)


def top_p_sampling(x, top_p=0.95):
    probs_sort, index = torch.sort(x, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    index = index[mask != 1]
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    probs = probs_sort[probs_sort != 0]
    return probs, index


class CNN_embedding(nn.Module):
    def __init__(self, config, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels=config.cnn_dim,
            out_channels=config.cnn_dim,
            kernel_size=7,
            padding=3,
            groups=config.cnn_dim,
        )
        self.norm = RMSNorm(config.cnn_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(config.cnn_dim, 4 * config.cnn_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * config.cnn_dim, config.cnn_dim)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((config.cnn_dim)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )
        self.downsample = nn.Conv2d(
            in_channels=config.channels,
            out_channels=config.cnn_dim,
            kernel_size=4,
            stride=4,
        )
        self.maxPool = nn.MaxPool2d(
            kernel_size=config.max_pool_size, stride=config.max_pool_stride
        )

    def forward(self, x):
        x = self.downsample(x)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 2, 1)
        x = input + x
        x = self.maxPool(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedd_dim = config.embedd_dim
        self.to_qkv = nn.Linear(
            config.embedd_dim, 3 * config.embedd_dim, bias=config.bias
        )
        self.scaling = config.embedd_dim ** (-0.5)
        self.heads = config.n_heads
        self.c_proj = nn.Linear(config.embedd_dim, config.embedd_dim, bias=config.bias)
        self.rotary_embed = RotaryEmbedding(config.embedd_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.to_qkv(x).split(self.embedd_dim, dim=-1)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)
        qk = q @ k.transpose(-1, -2)
        qk = qk * self.scaling
        attn = F.softmax(qk, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.factor = config.factor
        self.w1 = nn.Linear(
            config.embedd_dim, config.factor * config.embedd_dim, bias=config.bias
        )
        self.w2 = nn.Linear(
            config.factor * config.embedd_dim, config.embedd_dim, bias=config.bias
        )
        self.w3 = nn.Linear(
            config.embedd_dim, config.factor * config.embedd_dim, bias=config.bias
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class VitBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.ffn = FFN(config)
        self.attention_norm = RMSNorm(config.embedd_dim)
        self.ffn_norm = RMSNorm(config.embedd_dim)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.ffn(self.ffn_norm(h))
        return out


class RoutingVitBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.VitBlock = VitBlock(config)
        self.router = nn.Linear(config.embedd_dim, 1)
        self.top_k = config.top_k

    def forward(self, x):
        B, T, C = x.size()
        router_logits = self.router(x)
        probs, token_index = torch.topk(
            router_logits, k=self.top_k, dim=1, sorted=False
        )
        selected_tokens, index = torch.sort(token_index, dim=1)
        indices_expanded = selected_tokens.expand(-1, -1, self.config.embedd_dim)
        selected_tokens = torch.gather(x, 1, index=indices_expanded)
        output_attn = self.VitBlock(selected_tokens)
        tokens_weight = F.softmax(probs, dim=1)
        router_weights = torch.gather(tokens_weight, 1, index)
        output = router_weights * output_attn
        out = torch.scatter_add(input=x, dim=1, index=indices_expanded, src=output)
        return out


class RoutingVit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        image_height, image_width = pair(config.image_size)
        self.to_cnn_embedding = nn.Sequential(
            CNN_embedding(config, layer_scale_init_value=1e-6),
            Rearrange(
                "b c h w -> b (h w) c",
            ),
            RMSNorm(config.cnn_dim),
            nn.Linear(config.cnn_dim, config.embedd_dim, bias=config.bias),
            RMSNorm(config.embedd_dim),
        )
        self.pos_embedding = torch.nn.Parameter(
            torch.randn(1, config.num_patchs, config.embedd_dim)
        )

        self.layers = nn.ModuleList(
            [RoutingVitBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = RMSNorm(config.embedd_dim)
        self.to_logits = nn.Sequential(
            Reduce("b c h -> b h", "mean"),
            nn.Linear(config.embedd_dim, config.classes, bias=config.bias),
        )
        self.apply(self._init_weights)

    def _get_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.to_cnn_embedding(x)
        x += self.pos_embedding
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.to_logits(x)
        return logits

    def create_optimizer(self, learning_rate, weight_decay, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nondecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)

        print(f"decay params are {num_decay_params}")
        print(f"nondecay params are {num_nondecay_params}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"use fused AdamW; {use_fused}")
        return optimizer
