import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F


class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)

    @property
    def dtype(self):
        return self.prompt.dtype

    def forward(self, x):
        x = x[:, : self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x


class Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, alpha=1.0, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scaling = alpha / bottle_dim

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x * self.scaling


class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, alpha=1.0, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.alpha = alpha
        scaling_value = self.alpha / bottle_dim
        self.scaling = nn.Parameter(torch.ones(1, dtype=dtype) * scaling_value)

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x * self.scaling


class LoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, alpha=1, dtype=None):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_dim, bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, in_dim, dtype=dtype))
        self.alpha = alpha
        self.scaling = self.alpha / bottle_dim

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def dtype(self):
        return self.lora_A.dtype

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        return x * self.scaling


class MetaLoRA(LoRA):
    """LoRA with meta-parameter support for bi-level optimization"""

    def __init__(self, in_dim, bottle_dim, alpha=1.0, dtype=None, use_meta=False):
        super().__init__(in_dim, bottle_dim, alpha, dtype)
        self.use_meta = use_meta

        # Only create meta_alpha if meta-learning is enabled
        if use_meta:
            self.register_parameter("meta_alpha", nn.Parameter(torch.tensor(alpha, dtype=dtype)))
        else:
            self.meta_alpha = None

        # Inner loop state backup for bi-level optimization
        self.backup_lora_A = None
        self.backup_lora_B = None

    def forward(self, x):
        # Standard LoRA computation
        x = x @ self.lora_A
        x = x @ self.lora_B

        # Use meta_alpha or regular alpha based on mode
        if self.use_meta and self.meta_alpha is not None:
            scaling = self.meta_alpha / self.lora_A.size(1)
        else:
            scaling = self.alpha / self.lora_A.size(1)

        x = scaling * x
        return x

    def set_meta_mode(self, is_meta=False):
        """Switch between normal and meta mode"""
        self.use_meta = is_meta and self.meta_alpha is not None

    def get_meta_parameters(self):
        """Return meta parameters for optimization"""
        return [self.meta_alpha] if self.meta_alpha is not None else []

    def backup_weights(self):
        """Save current weights for later restoration"""
        self.backup_lora_A = self.lora_A.data.clone()
        self.backup_lora_B = self.lora_B.data.clone()

    def restore_weights(self):
        """Restore weights from backup"""
        if self.backup_lora_A is not None and self.backup_lora_B is not None:
            self.lora_A.data.copy_(self.backup_lora_A)
            self.lora_B.data.copy_(self.backup_lora_B)


class MLPLoRA(LoRA):
    def __init__(self, in_dim, bottle_dim, out_dim, alpha=1, dtype=None):
        super().__init__(in_dim, bottle_dim, alpha=alpha, dtype=dtype)
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, out_dim, dtype=dtype))
        nn.init.zeros_(self.lora_B)


class FLoRA(nn.Module):
    """
    Fine-grained LoRA provides control over position-wise rank, alpha, and lr.

    Args:
        in_dim (int): Input dimension.
        rank (int): Rank of the low-rank approximation.
        out_dim (int, optional): Output dimension. If None, it's set to in_dim. Default: None.
        dtype (torch.dtype, optional): Data type of the parameters. Default: None.
    """

    def __init__(self, in_dim, rank, alpha=None, out_dim=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.out_dim = out_dim if out_dim is not None else in_dim

        self.lora_A = nn.Parameter(torch.zeros(self.in_dim, self.rank, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_dim, dtype=dtype))
        self.alpha = rank if alpha is None else alpha
        # alpha controls the magnitude of LoRA updates relative to
        # the original weights (higher alpha -> stronger adaptation)
        # Usually alpha is set to rank, i.e., scaling = 1
        # If alpha > rank, scaling > 1, i.e., stronger adaptation
        # If alpha < rank, scaling < 1, i.e., weaker adaptation
        self.scaling = self.alpha / self.rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def dtype(self):
        return self.lora_A.dtype

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
        return x


class FDoRA(nn.Module):
    """
    Enhanced LoRA with multiple diagonal matrices for fine-grained control.

    Args:
        in_dim (int): Input dimension
        rank (int): Rank of the low-rank approximation
        num_D (int): Number of diagonal matrices to use
        out_dim (int, optional): Output dimension. If None, set to in_dim
        dtype (torch.dtype, optional): Data type of the parameters
    """

    def __init__(self, in_dim, rank, num_D=1, out_dim=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.out_dim = out_dim if out_dim is not None else in_dim

        # 基础LoRA参数
        self.lora_A = nn.Parameter(torch.zeros(self.in_dim, self.rank, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_dim, dtype=dtype))

        # 多个对角矩阵D (D1D2...Dk)
        self.D_matrices = nn.ParameterList([nn.Parameter(torch.ones(rank, dtype=dtype)) for _ in range(num_D)])

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # x → A
        x = x @ self.lora_A  # [batch_size, in_dim] @ [in_dim, rank] = [batch_size, rank]

        # 应用所有D矩阵
        # 注意：由于D是对角矩阵，我们只需要存储对角线元素并做element-wise乘法
        for D in self.D_matrices:
            x = x * D.unsqueeze(0)  # [batch_size, rank] * [1, rank]

        # → B
        x = x @ self.lora_B  # [batch_size, rank] @ [rank, out_dim] = [batch_size, out_dim]

        return x


class FLoTR(FLoRA):
    """
    Fine-grained LoTR (Low Tensor Rank adaptation) that inherits from FLoRA
    and extends it with tensor decomposition across multiple layers.

    Args:
        in_dim (int): Input dimension
        rank (int): Rank of the tensor decomposition
        num_layers (int): Number of layers to adapt
        alpha (float, optional): Scaling factor. If None, defaults to rank
        out_dim (int, optional): Output dimension. If None, defaults to in_dim
        dtype (torch.dtype, optional): Data type of the parameters
    """

    def __init__(self, in_dim, rank, num_layers, alpha=None, out_dim=None, dtype=None):
        # Initialize parent FLoRA class
        super().__init__(in_dim=in_dim, rank=rank, alpha=alpha, out_dim=out_dim, dtype=dtype)

        self.num_layers = num_layers

        # Rename LoRA parameters to match LoTR terminology
        # self.lora_A and self.lora_B are inherited from FLoRA

        # Add core tensor G specific to LoTR
        self.core_G = nn.Parameter(torch.zeros(self.rank, self.num_layers, self.rank, dtype=dtype))

        # Initialize core tensor with zeros as per paper
        nn.init.zeros_(self.core_G)

        # Track current layer
        self.current_layer = 0

    def set_layer_idx(self, idx: int):
        """Set the current layer index for forward pass."""
        # WARN: need to set layer index before forward pass
        assert 0 <= idx < self.num_layers, f"Layer index {idx} out of range [0, {self.num_layers})"
        self.current_layer = idx

    def forward(self, x):
        """
        Forward pass incorporating tensor decomposition.
        Overrides FLoRA's forward pass.

        Args:
            x: Input tensor of shape (*, in_dim)

        Returns:
            Tensor of shape (*, out_dim)
        """
        # Get layer-specific core matrix
        G_s = self.core_G[:, self.current_layer, :]  # Shape: (rank, rank)

        # Compute adaptation: scaling * (B @ G_s @ A.T) @ x.T
        x = x @ self.lora_A  # Shape: (*, rank)
        x = x @ G_s.T  # Shape: (*, rank)
        x = x @ self.lora_B.T  # Shape: (*, out_dim)
        return self.scaling * x


class SSF(nn.Module):
    def __init__(self, in_dim, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(in_dim, dtype=dtype))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    @property
    def dtype(self):
        return self.scale.dtype

    def forward(self, x):
        if len(x.shape) == 4:  # for CNN
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        else:
            return x * self.scale + self.shift


class MaskedLinear(nn.Module):
    def __init__(self, weight, bias, ratio=0.0, generator=None):
        super().__init__()
        # weight: (out_dim, in_dim)
        # bias: (out_dim)
        out_dim, in_dim = weight.shape
        num_params = out_dim * in_dim + out_dim
        ratio = float(eval(ratio)) if isinstance(ratio, str) else float(ratio)
        num_masked = int(num_params * ratio)

        # randomly select the optimized parameters
        masked_indexs = torch.randperm(num_params, generator=generator)[:num_masked]
        mask = torch.zeros(num_params, dtype=bool).scatter(dim=0, index=masked_indexs, value=True)
        mask = mask.reshape(out_dim, in_dim + 1)
        self.mask_weight = mask[:, :-1]
        self.mask_bias = mask[:, -1]

        self.optimized_weight = nn.Parameter(torch.masked_select(weight.detach(), mask=self.mask_weight))
        self.optimized_bias = nn.Parameter(torch.masked_select(bias.detach(), mask=self.mask_bias))

    def forward(self, x, weight, bias):
        self.mask_weight = self.mask_weight.to(weight.device)
        self.mask_bias = self.mask_bias.to(bias.device)

        if self.mask_weight.sum() > 0:
            weight = torch.masked_scatter(weight, mask=self.mask_weight, source=self.optimized_weight)
        if self.mask_bias.sum() > 0:
            bias = torch.masked_scatter(bias, mask=self.mask_bias, source=self.optimized_bias)
        return F.linear(x, weight, bias)


class MetaAdapter(Adapter):
    """Adapter with meta-parameter support for bi-level optimization"""

    def __init__(self, in_dim, bottle_dim, alpha=1.0, dtype=None, use_meta=False):
        super().__init__(in_dim, bottle_dim, alpha=alpha, dtype=dtype)
        self.use_meta = use_meta
        # Create per-feature meta scale for more expressivity
        if use_meta:
            self.register_parameter("meta_scale", nn.Parameter(torch.ones(in_dim, dtype=dtype)))
        else:
            self.meta_scale = None

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)

        # Use meta_scale or regular scaling based on mode
        if self.use_meta and self.meta_scale is not None:
            return x * self.meta_scale.unsqueeze(0)  # Broadcasting for batch dimension
        else:
            return x * self.scaling

    def set_meta_mode(self, is_meta=False):
        """Switch between normal and meta mode"""
        self.use_meta = is_meta and self.meta_scale is not None

    def get_meta_parameters(self):
        """Return meta parameters for optimization"""
        return [self.meta_scale] if self.meta_scale is not None else []

    def backup_weights(self):
        """Save current weights for later restoration"""
        self.backup_down_proj = self.down_proj.weight.data.clone()
        self.backup_down_bias = self.down_proj.bias.data.clone()
        self.backup_up_proj = self.up_proj.weight.data.clone()
        self.backup_up_bias = self.up_proj.bias.data.clone()
        self.backup_ln_weight = self.ln.weight.data.clone()
        self.backup_ln_bias = self.ln.bias.data.clone()

    def restore_weights(self):
        """Restore weights from backup"""
        if hasattr(self, "backup_down_proj"):
            self.down_proj.weight.data.copy_(self.backup_down_proj)
            self.down_proj.bias.data.copy_(self.backup_down_bias)
            self.up_proj.weight.data.copy_(self.backup_up_proj)
            self.up_proj.bias.data.copy_(self.backup_up_bias)
            self.ln.weight.data.copy_(self.backup_ln_weight)
            self.ln.bias.data.copy_(self.backup_ln_bias)


class MetaAdaptFormer(AdaptFormer):
    """AdaptFormer with meta-parameter support for bi-level optimization"""

    def __init__(self, in_dim, bottle_dim, alpha=1.0, dtype=None, use_meta=False):
        super().__init__(in_dim, bottle_dim, alpha=alpha, dtype=dtype)
        self.use_meta = use_meta

        # Create meta_scale parameter - initialize to ones
        if use_meta:
            # Change to vector parameter for more expressive power
            self.register_parameter("meta_scale", nn.Parameter(torch.ones(in_dim, dtype=dtype)))
        else:
            self.meta_scale = None

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)

        # Use meta_scale or regular scaling based on mode
        if self.use_meta and self.meta_scale is not None:
            return x * self.meta_scale.unsqueeze(0)
        else:
            return x * self.scaling

    def set_meta_mode(self, is_meta=False):
        """Switch between normal and meta mode"""
        self.use_meta = is_meta and self.meta_scale is not None

    def get_meta_parameters(self):
        """Return meta parameters for optimization"""
        return [self.meta_scale] if self.meta_scale is not None else []

    def backup_weights(self):
        """Save current weights for later restoration"""
        self.backup_down_proj = self.down_proj.weight.data.clone()
        self.backup_down_bias = self.down_proj.bias.data.clone()
        self.backup_up_proj = self.up_proj.weight.data.clone()
        self.backup_up_bias = self.up_proj.bias.data.clone()
        self.backup_ln_weight = self.ln.weight.data.clone()
        self.backup_ln_bias = self.ln.bias.data.clone()
        self.backup_scale = self.scaling.data.clone()

    def restore_weights(self):
        """Restore weights from backup"""
        if hasattr(self, "backup_down_proj"):
            self.down_proj.weight.data.copy_(self.backup_down_proj)
            self.down_proj.bias.data.copy_(self.backup_down_bias)
            self.up_proj.weight.data.copy_(self.backup_up_proj)
            self.up_proj.bias.data.copy_(self.backup_up_bias)
            self.ln.weight.data.copy_(self.backup_ln_weight)
            self.ln.bias.data.copy_(self.backup_ln_bias)
            self.scaling.data.copy_(self.backup_scale)
