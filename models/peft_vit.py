import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as ViT

from clip.model import VisionTransformer as CLIP_ViT
from models.satmae_vit import MAEViTAdapter

from .peft_modules import (
    SSF,
    VPT,
    Adapter,
    AdaptFormer,
    FLoRA,
    MaskedLinear,
    MLPLoRA,
    MetaLoRA,
    MetaAdapter,
    MetaAdaptFormer,
)


class ViT_Tuner(nn.Module):
    """All instance variables in this class will be optimized."""

    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()

        if isinstance(vit_model, CLIP_ViT):
            n_layers = len(vit_model.transformer.resblocks)
            emb_dim = vit_model.positional_embedding.shape[1]
            seq_len = vit_model.positional_embedding.shape[0]
            patch_size = vit_model.conv1.kernel_size
            dtype = vit_model.conv1.weight.dtype

            blocks = vit_model.transformer.resblocks

            get_attn_in_weight = lambda i: blocks[i].attn.in_proj_weight
            get_attn_in_bias = lambda i: blocks[i].attn.in_proj_bias
            get_attn_out_weight = lambda i: blocks[i].attn.out_proj.weight
            get_attn_out_bias = lambda i: blocks[i].attn.out_proj.bias
            get_mlp_in_weight = lambda i: blocks[i].mlp[0].weight
            get_mlp_in_bias = lambda i: blocks[i].mlp[0].bias
            get_mlp_out_weight = lambda i: blocks[i].mlp[2].weight
            get_mlp_out_bias = lambda i: blocks[i].mlp[2].bias

            attn_in_dim = get_attn_in_bias(0).shape[0]
            attn_out_dim = get_attn_out_bias(0).shape[0]
            mlp_in_dim = get_mlp_in_bias(0).shape[0]
            mlp_out_dim = get_mlp_out_bias(0).shape[0]

        elif isinstance(vit_model, ViT) or isinstance(vit_model, MAEViTAdapter):
            n_layers = len(vit_model.blocks)
            emb_dim = vit_model.pos_embed.shape[2]
            seq_len = vit_model.pos_embed.shape[1]
            patch_size = vit_model.patch_embed.proj.kernel_size
            dtype = vit_model.patch_embed.proj.weight.dtype

            blocks = vit_model.blocks

            get_attn_in_weight = lambda i: blocks[i].attn.qkv.weight
            get_attn_in_bias = lambda i: blocks[i].attn.qkv.bias
            get_attn_out_weight = lambda i: blocks[i].attn.proj.weight
            get_attn_out_bias = lambda i: blocks[i].attn.proj.bias
            get_mlp_in_weight = lambda i: blocks[i].mlp.fc1.weight
            get_mlp_in_bias = lambda i: blocks[i].mlp.fc1.bias
            get_mlp_out_weight = lambda i: blocks[i].mlp.fc2.weight
            get_mlp_out_bias = lambda i: blocks[i].mlp.fc2.bias

            attn_in_dim = get_attn_in_bias(0).shape[0]
            attn_out_dim = get_attn_out_bias(0).shape[0]
            mlp_in_dim = get_mlp_in_bias(0).shape[0]
            mlp_out_dim = get_mlp_out_bias(0).shape[0]

        use_full_tuning = cfg.full_tuning
        use_bias_tuning = cfg.bias_tuning
        use_ln_tuning = cfg.ln_tuning
        use_vpt_shallow = cfg.vpt_shallow
        use_vpt_deep = cfg.vpt_deep
        use_adapter = cfg.adapter
        use_adaptformer = cfg.adaptformer
        use_lora = cfg.lora
        use_lora_mlp = cfg.lora_mlp
        scale_alpha = cfg.scale_alpha
        use_flora = cfg.use_flora
        use_meta = cfg.use_meta
        use_ssf_attn = cfg.ssf_attn
        use_ssf_mlp = cfg.ssf_mlp
        use_ssf_ln = cfg.ssf_ln
        use_mask = cfg.mask
        partial = cfg.partial
        vpt_len = cfg.vpt_len
        adapter_dim = cfg.adapter_dim
        mask_ratio = cfg.mask_ratio
        mask_seed = cfg.mask_seed

        if partial is None:
            _start, _end = 0, n_layers
        elif isinstance(partial, int):
            _start, _end = n_layers - partial, n_layers
            print(f"Partial tuning: Tuning last {partial} layers (layers {_start} to {_end-1})")
        elif isinstance(partial, list):
            _start, _end = partial[0], partial[1]
            print(f"Partial tuning: Tuning layers {_start} to {_end-1}")

        if (use_vpt_shallow or use_vpt_deep) and (vpt_len is None):
            vpt_len = 10
            print(f"Visual prompt length set to {vpt_len}")

        if use_adapter or use_adaptformer or use_lora or use_lora_mlp or use_flora:
            if adapter_dim is None:
                adapter_dim = 2 ** max(0, int(math.log2(num_classes / (n_layers * 2))))
                print(f"Adapter bottle dimension set to {adapter_dim}")
                # adapter_dim = max(1, num_classes // (n_layers * 2))
            if scale_alpha == "rank":
                scale_alpha = adapter_dim
                print(f"Scaling alpha set to adapter dimension {adapter_dim}")
            else:
                print(f"Using scaling alpha {scale_alpha}")

        if use_mask and mask_ratio is None:
            mask_ratio = num_classes / (12 * n_layers * emb_dim)
            mask_ratio = max(0.001, mask_ratio // 0.001 * 0.001)
            print(f"Mask ratio set to {mask_ratio}")

        if use_mask and mask_seed is None:
            mask_seed = 0
            print(f"Mask seed set to {mask_seed}")

        if use_full_tuning:
            block_tuned = blocks[_start:_end]
            print(f"Full tuning enabled for layers {_start} to {_end-1}")
        else:
            block_tuned = None

        if use_bias_tuning:
            bias_tuned = nn.ParameterList([param for name, param in blocks.named_parameters() if name.endswith("bias")])
            print("Bias tuning enabled")
        else:
            bias_tuned = None

        if use_ln_tuning:
            ln_tuned = nn.ModuleList([mod for name, mod in blocks.named_modules() if isinstance(mod, nn.LayerNorm)])
            print("Layer norm tuning enabled")
        else:
            ln_tuned = None

        assert bool(use_vpt_shallow) + bool(use_vpt_deep) < 2
        if use_vpt_shallow:
            vpt_list = nn.ModuleList(
                [
                    VPT(
                        vpt_len=vpt_len,
                        seq_len=seq_len,
                        patch_size=patch_size,
                        emb_dim=emb_dim,
                        dtype=dtype,
                    ),
                    *[None] * (n_layers - 1),
                ]
            )
            print(f"Shallow VPT enabled with length {vpt_len}")
        elif use_vpt_deep:
            vpt_list = nn.ModuleList(
                [
                    *[None] * (_start),
                    *[
                        VPT(
                            vpt_len=vpt_len,
                            seq_len=seq_len,
                            patch_size=patch_size,
                            emb_dim=emb_dim,
                            dtype=dtype,
                        )
                        for _ in range(_start, _end)
                    ],
                    *[None] * (n_layers - _end),
                ]
            )
            print(f"Deep VPT enabled with length {vpt_len} for layers {_start} to {_end-1}")
        else:
            vpt_list = nn.ModuleList([None] * n_layers)

        if use_adapter:
            adapter_list = nn.ModuleList(
                [
                    *[None] * (_start),
                    *[
                        (
                            MetaAdapter(in_dim=emb_dim, bottle_dim=adapter_dim, alpha=scale_alpha, dtype=dtype, use_meta=use_meta)
                            if use_meta
                            else Adapter(in_dim=emb_dim, bottle_dim=adapter_dim, alpha=scale_alpha, dtype=dtype)
                        )
                        for _ in range(_start, _end)
                    ],
                    *[None] * (n_layers - _end),
                ]
            )
            print(f"{'Meta' if use_meta else ''}Adapter enabled with dimension {adapter_dim} for layers {_start} to {_end-1}")
        else:
            adapter_list = nn.ModuleList([None] * n_layers)

        if use_adaptformer:
            adaptformer_list = nn.ModuleList(
                [
                    *[None] * (_start),
                    *[
                        (
                            MetaAdaptFormer(in_dim=emb_dim, bottle_dim=adapter_dim, alpha=scale_alpha, dtype=dtype, use_meta=use_meta)
                            if use_meta
                            else AdaptFormer(in_dim=emb_dim, bottle_dim=adapter_dim, alpha=scale_alpha, dtype=dtype)
                        )
                        for _ in range(_start, _end)
                    ],
                    *[None] * (n_layers - _end),
                ]
            )
            print(f"{'Meta' if use_meta else ''}AdaptFormer enabled with dimension {adapter_dim} for layers {_start} to {_end-1}")
        else:
            adaptformer_list = nn.ModuleList([None] * n_layers)

        if use_lora:
            lora_list = nn.ModuleList(
                [
                    *[None] * (_start),
                    *[
                        nn.ModuleDict(
                            {
                                "q": MetaLoRA(
                                    in_dim=emb_dim,
                                    bottle_dim=adapter_dim,
                                    alpha=scale_alpha,
                                    dtype=dtype,
                                    use_meta=use_meta,
                                ),
                                "v": MetaLoRA(
                                    in_dim=emb_dim,
                                    bottle_dim=adapter_dim,
                                    alpha=scale_alpha,
                                    dtype=dtype,
                                    use_meta=use_meta,
                                ),
                            }
                        )
                        for _ in range(_start, _end)
                    ],
                    *[None] * (n_layers - _end),
                ]
            )
            print(f"{'Meta' if use_meta else ''}LoRA enabled with dimension {adapter_dim} for layers {_start} to {_end-1}")
        else:
            lora_list = nn.ModuleList([None] * n_layers)

        if use_lora_mlp:
            lora_mlp_list = nn.ModuleList(
                [
                    *[None] * (_start),
                    *[
                        nn.ModuleDict(
                            {
                                "1": MLPLoRA(
                                    in_dim=emb_dim,
                                    bottle_dim=adapter_dim,
                                    out_dim=mlp_in_dim,
                                    alpha=scale_alpha,
                                    dtype=dtype,
                                ),
                                "2": MLPLoRA(
                                    in_dim=mlp_in_dim,
                                    bottle_dim=adapter_dim,
                                    out_dim=emb_dim,
                                    alpha=scale_alpha,
                                    dtype=dtype,
                                ),
                            }
                        )
                        for _ in range(_start, _end)
                    ],
                    *[None] * (n_layers - _end),
                ]
            )
            print(f"LoRA MLP enabled with dimension {adapter_dim} for layers {_start} to {_end-1}")
        else:
            lora_mlp_list = nn.ModuleList([None] * n_layers)

        if use_flora:
            flora_arch = cfg.flora.arch
            # Initialize an empty list of FLoRA modules for each layer
            flora_list = [None] * n_layers

            # Create FLoRA modules for specified layers
            for layer_idx in flora_arch.layers:
                flora_modules = {}
                for module_name in flora_arch.modules:
                    if module_name in ["q", "k", "v", "out"]:
                        flora_modules[module_name] = FLoRA(
                            in_dim=emb_dim,
                            rank=flora_arch.rank,
                            alpha=flora_arch.alpha,
                            dtype=dtype,
                        )
                    elif module_name == "mlp1":
                        flora_modules[module_name] = FLoRA(
                            in_dim=emb_dim,
                            rank=flora_arch.rank,
                            alpha=flora_arch.alpha,
                            out_dim=mlp_in_dim,
                            dtype=dtype,
                        )
                    elif module_name == "mlp2":
                        flora_modules[module_name] = FLoRA(
                            in_dim=mlp_in_dim,
                            rank=flora_arch.rank,
                            alpha=flora_arch.alpha,
                            out_dim=emb_dim,
                            dtype=dtype,
                        )
                    else:
                        raise ValueError(f"Invalid module name: {module_name}")
                if flora_modules:
                    flora_list[layer_idx] = nn.ModuleDict(flora_modules)

            flora_list = nn.ModuleList(flora_list)
            print(f"FLoRA initialized with rank {flora_arch.rank} for modules {flora_arch.modules}")
        else:
            flora_list = nn.ModuleList([None] * n_layers)
            print("FLoRA not used.")

        if use_ssf_attn:
            ssf_attn_list = nn.ModuleList(
                [
                    *[None] * (_start),
                    *[
                        nn.ModuleDict(
                            {
                                "attn_in": SSF(attn_in_dim, dtype=dtype),
                                "attn_out": SSF(attn_out_dim, dtype=dtype),
                            }
                        )
                        for _ in range(_start, _end)
                    ],
                    *[None] * (n_layers - _end),
                ]
            )
            print(f"SSF Attention enabled for layers {_start} to {_end-1}")
        else:
            ssf_attn_list = nn.ModuleList([None] * n_layers)

        if use_ssf_mlp:
            ssf_mlp_list = nn.ModuleList(
                [
                    *[None] * (_start),
                    *[
                        nn.ModuleDict(
                            {
                                "mlp_in": SSF(mlp_in_dim, dtype=dtype),
                                "mlp_out": SSF(mlp_out_dim, dtype=dtype),
                            }
                        )
                        for _ in range(_start, _end)
                    ],
                    *[None] * (n_layers - _end),
                ]
            )
            print(f"SSF MLP enabled for layers {_start} to {_end-1}")
        else:
            ssf_mlp_list = nn.ModuleList([None] * n_layers)

        if use_ssf_ln:
            ssf_ln_list = nn.ModuleList(
                [
                    *[None] * (_start),
                    *[
                        nn.ModuleDict(
                            {
                                "ln_1": SSF(emb_dim, dtype=dtype),
                                "ln_2": SSF(emb_dim, dtype=dtype),
                            }
                        )
                        for _ in range(_start, _end)
                    ],
                    *[None] * (n_layers - _end),
                ]
            )
            print(f"SSF Layer Norm enabled for layers {_start} to {_end-1}")
        else:
            ssf_ln_list = nn.ModuleList([None] * n_layers)

        if use_mask:
            generator = torch.Generator().manual_seed(mask_seed)
            masked_linear_list = nn.ModuleList(
                [
                    *[None] * (_start),
                    *[
                        nn.ModuleDict(
                            {
                                "attn_in": MaskedLinear(
                                    weight=get_attn_in_weight(i),
                                    bias=get_attn_in_bias(i),
                                    ratio=mask_ratio,
                                    generator=generator,
                                ),
                                "attn_out": MaskedLinear(
                                    weight=get_attn_out_weight(i),
                                    bias=get_attn_out_bias(i),
                                    ratio=mask_ratio,
                                    generator=generator,
                                ),
                                "mlp_in": MaskedLinear(
                                    weight=get_mlp_in_weight(i),
                                    bias=get_mlp_in_bias(i),
                                    ratio=mask_ratio,
                                    generator=generator,
                                ),
                                "mlp_out": MaskedLinear(
                                    weight=get_mlp_out_weight(i),
                                    bias=get_mlp_out_bias(i),
                                    ratio=mask_ratio,
                                    generator=generator,
                                ),
                            }
                        )
                        for i in range(_start, _end)
                    ],
                    *[None] * (n_layers - _end),
                ]
            )
            print(f"Masked Linear enabled with ratio {mask_ratio} and seed {mask_seed} for layers {_start} to {_end-1}")
        else:
            masked_linear_list = nn.ModuleList([None] * n_layers)

        # To be optimized
        self.block_tuned = block_tuned
        self.bias_tuned = bias_tuned
        self.ln_tuned = ln_tuned
        self.vpt_list = vpt_list
        self.adapter_list = adapter_list
        self.adaptformer_list = adaptformer_list
        self.lora_list = lora_list
        self.lora_mlp_list = lora_mlp_list
        self.ssf_attn_list = ssf_attn_list
        self.ssf_mlp_list = ssf_mlp_list
        self.ssf_ln_list = ssf_ln_list
        self.masked_linear_list = masked_linear_list
        self.flora_list = flora_list

        print("ViT_Tuner initialization complete")


class Peft_ViT(nn.Module):
    def __init__(self, vit_model):
        super().__init__()

        if isinstance(vit_model, CLIP_ViT):
            self.backbone = "CLIP-VIT"
            self.patch_embedding = vit_model.conv1
            self.class_embedding = vit_model.class_embedding
            self.positional_embedding = vit_model.positional_embedding
            self.ln_pre = vit_model.ln_pre
            self.blocks = vit_model.transformer.resblocks
            self.ln_post = vit_model.ln_post
            self.proj = vit_model.proj  # not used
            self.out_dim = self.ln_post.bias.shape[0]
            # self.out_dim = self.proj.shape[1]

        elif isinstance(vit_model, ViT):
            self.backbone = "ViT"
            self.patch_embedding = vit_model.patch_embed.proj
            self.class_embedding = vit_model.cls_token
            self.positional_embedding = vit_model.pos_embed
            self.ln_pre = vit_model.norm_pre
            self.blocks = vit_model.blocks
            self.ln_post = vit_model.norm
            self.proj = nn.Identity()
            self.out_dim = self.ln_post.bias.shape[0]

        elif isinstance(vit_model, MAEViTAdapter):
            self.backbone = "ViT"
            self.patch_embedding = vit_model.patch_embed.proj
            self.class_embedding = vit_model.cls_token
            self.positional_embedding = vit_model.pos_embed
            self.ln_pre = vit_model.norm_pre if hasattr(vit_model, "norm_pre") else nn.Identity()
            self.blocks = vit_model.blocks
            if hasattr(vit_model, "fc_norm"):
                self.ln_post = vit_model.fc_norm
            else:
                self.ln_post = vit_model.norm
            self.proj = nn.Identity()
            self.out_dim = self.ln_post.bias.shape[0]

    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    def forward(self, x, tuner=None, head=None):
        x = x.to(self.dtype)
        x = self.patch_embedding(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        _bsz = x.shape[0]
        _ = x.shape[1]  # unused: _seg_len
        _emb_dim = x.shape[2]

        n_layers = len(self.blocks)

        for i in range(n_layers):
            block = self.blocks[i]

            if tuner is not None:
                vpt = tuner.vpt_list[i]
                adapter = tuner.adapter_list[i]
                adaptformer = tuner.adaptformer_list[i]
                lora = tuner.lora_list[i]
                lora_mlp = tuner.lora_mlp_list[i]
                ssf_attn = tuner.ssf_attn_list[i]
                ssf_mlp = tuner.ssf_mlp_list[i]
                ssf_ln = tuner.ssf_ln_list[i]
                masked_linear = tuner.masked_linear_list[i]
                flora = tuner.flora_list[i]
            else:
                vpt = adapter = adaptformer = lora = lora_mlp = ssf_attn = ssf_mlp = ssf_ln = flora = masked_linear = None

            if vpt is not None:
                x = vpt(x)

            _seq_len_after_vpt = x.shape[1]

            x = x.permute(1, 0, 2)  # NLD -> LND

            if self.backbone == "CLIP-VIT":
                _attn = block.attn
                _ln_1 = block.ln_1
                _mlp = block.mlp
                _ln_2 = block.ln_2

                _attn_in_proj_weight = _attn.in_proj_weight
                _attn_in_proj_bias = _attn.in_proj_bias
                _attn_out_proj_weight = _attn.out_proj.weight
                _attn_out_proj_bias = _attn.out_proj.bias
                _mlp_in_proj_weight = _mlp[0].weight
                _mlp_in_proj_bias = _mlp[0].bias
                _mlp_act = _mlp[1]
                _mlp_out_proj_weight = _mlp[2].weight
                _mlp_out_proj_bias = _mlp[2].bias

                _num_heads = _attn.num_heads
                _head_dim = _emb_dim // _num_heads

            elif self.backbone == "ViT":
                _attn = block.attn
                _ln_1 = block.norm1
                _mlp = block.mlp
                _ln_2 = block.norm2

                _attn_in_proj_weight = _attn.qkv.weight
                _attn_in_proj_bias = _attn.qkv.bias
                _attn_out_proj_weight = _attn.proj.weight
                _attn_out_proj_bias = _attn.proj.bias
                _mlp_in_proj_weight = _mlp.fc1.weight
                _mlp_in_proj_bias = _mlp.fc1.bias
                _mlp_act = _mlp.act
                _mlp_out_proj_weight = _mlp.fc2.weight
                _mlp_out_proj_bias = _mlp.fc2.bias

                _num_heads = _attn.num_heads
                _head_dim = _emb_dim // _num_heads

            ###############################
            #  Multi-Head Self-Attention  #
            ###############################
            identity = x  # deep copy

            x = _ln_1(x)
            if ssf_ln is not None:
                x = ssf_ln["ln_1"](x)

            if masked_linear is not None:
                qkv = masked_linear["attn_in"](x, _attn_in_proj_weight, _attn_in_proj_bias)
            else:
                qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            if lora is not None:
                q = q + lora["q"](x)
                v = v + lora["v"](x)

            if flora is not None:
                if "q" in flora:
                    q = q + flora["q"](x)
                if "k" in flora:
                    k = k + flora["k"](x)
                if "v" in flora:
                    v = v + flora["v"](x)

            if ssf_attn is not None:
                qkv = torch.cat([q, k, v], dim=-1)
                qkv = ssf_attn["attn_in"](qkv)
                q, k, v = qkv.chunk(3, dim=-1)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)

            x = F.scaled_dot_product_attention(q, k, v)
            # scaled_dot_product_attention:
            # q = q / math.sqrt(_head_dim)
            # attn = torch.bmm(q, k.transpose(-2, -1))
            # attn = F.softmax(attn, dim=-1)
            # x = torch.bmm(attn, v)

            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)

            if masked_linear is not None:
                x = masked_linear["attn_out"](x, _attn_out_proj_weight, _attn_out_proj_bias)
            else:
                x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)

            if flora is not None and "out" in flora:
                x = x + flora["out"](x)

            if ssf_attn is not None:
                x = ssf_attn["attn_out"](x)

            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)

            x = x + identity

            ##########################
            #  Feed-Forward Network  #
            ##########################
            identity = x  # deep copy

            x = _ln_2(x)
            if ssf_ln is not None:
                x = ssf_ln["ln_2"](x)

            if masked_linear is not None:
                x_out = masked_linear["mlp_in"](x, _mlp_in_proj_weight, _mlp_in_proj_bias)
            else:
                x_out = F.linear(x, _mlp_in_proj_weight, _mlp_in_proj_bias)

            if lora_mlp is not None:
                x_out = x_out + lora_mlp["1"](x)

            if flora is not None and "mlp1" in flora:
                x_out = x_out + flora["mlp1"](x)

            x = x_out

            if ssf_mlp is not None:
                x = ssf_mlp["mlp_in"](x)

            x = _mlp_act(x)

            if masked_linear is not None:
                x_out = masked_linear["mlp_out"](x, _mlp_out_proj_weight, _mlp_out_proj_bias)
            else:
                x_out = F.linear(x, _mlp_out_proj_weight, _mlp_out_proj_bias)

            if lora_mlp is not None:
                x_out = x_out + lora_mlp["2"](x)

            if flora is not None and "mlp2" in flora:
                x_out = x_out + flora["mlp2"](x)

            x = x_out

            if ssf_mlp is not None:
                x = ssf_mlp["mlp_out"](x)

            if adapter is not None:
                x = x + adapter(x)

            if adaptformer is not None:
                x = x + adaptformer(identity)

            x = x + identity

            x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 0, :]  # extract only the cls token
        x = self.ln_post(x)  # apply layer norm
        # x = x @ self.proj

        if head is None:
            return x
        else:
            return head(x)
