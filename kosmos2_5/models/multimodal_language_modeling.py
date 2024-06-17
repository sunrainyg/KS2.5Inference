# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq import distributed_utils, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, Embedding
from .modules.positional_embedding import PositionalEmbedding
from omegaconf import II

from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder

from .beit2 import modeling_vqkd
from timm.models import create_model


DEFAULT_MAX_TARGET_POSITIONS = 1024
logger = logging.getLogger(__name__)


@dataclass
class MultimodalLanguageConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropoutattention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        },
    )
    moe_freq: int = field(
        default=0,
        metadata={"help": "Frequency at which we insert MoE Transformer layers"},
    )
    moe_expert_count: int = field(
        default=0, metadata={"help": "Number of experts in each MoE Layer"}
    )
    moe_gating_use_fp32: bool = field(
        default=False,
        metadata={"help": "Use FP32 computations in MoE top2 gating function"},
    )
    moe_second_expert_policy: str = field(
        default="sampling",
        metadata={"help": "policy for second expert, options: all/sampling/random"},
    )
    moe_normalize_gate_prob_before_dropping: bool = field(
        default=False,
        metadata={
            "help": "whether to normalize gate probs before or after dropping experts for capacity and randomization"
        },
    )
    moe_expert_ffn_dim: Optional[int] = field(
        default=None, metadata={"help": "MoE expert FFN dimension"}
    )
    moe_top1_expert: Optional[bool] = field(
        default=False, metadata={"help": "Use top1 gate instead of top2"}
    )
    moe_eval_capacity_token_fraction: Optional[float] = field(
        default=0.25,
        metadata={
            "help": (
                "Default: 0.25, Fraction of tokens as capacity during validation, "
                "if set to negative, use same as training. range: (0.0, 1.0]."
            )
        },
    )
    moe_normalize_expert_grad: Optional[str] = field(
        default="world_size",
        metadata={
            "help": "Divide expert gradients by (1) 'world_size' (2) 'sqrt_world_size'"
        },
    )
    record_a2a_perf_stats: Optional[bool] = field(
        default=False,
        metadata={"help": "records all to all perf stats during distributed training"},
    )
    dummy_a2a: Optional[bool] = field(
        default=False,
        metadata={
            "help": "By passes all to all during distributed training by returning the input buffer as output"
        },
    )
    moe_batch_prioritized_routing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if true orders token by the gate prob before capacity dropping."
        },
    )
    use_xmoe: Optional[bool] = field(
        default=False,
    )
    flash_attention: Optional[bool] = field(
        default=True,
    )
    sope_rel_pos: Optional[bool] = field(
        default=False,
        metadata={"help": "use SoPE as the relative position embhedding"},
    )
    xpos_rel_pos: Optional[bool] = field(
        default=False
    )
    scale_length: Optional[int] = field(
        default=2048,
    )

    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    fp16: bool = II("common.fp16")
    fp16_no_flatten_grads: bool = II("common.fp16_no_flatten_grads")
    ddp_backend: str = II("distributed_training.ddp_backend")
    world_size: int = II("distributed_training.distributed_world_size")
    distributed_rank: int = II("distributed_training.distributed_rank")
    ddp_rank: int = II("distributed_training.distributed_rank")
    deepnorm: Optional[bool] = field(
        default=False,
    )
    subln: Optional[bool] = field(
        default=False,
    )
    rel_pos_buckets: Optional[int] = field(
        default=0,
    )
    max_rel_pos: Optional[int] = field(
        default=0,
    )
    
    input_bits: Optional[int] = field(
        default=8,
    )
    input_quant_method: Optional[str] = field(
        default='elastic',
    )
    weight_bits: Optional[int] = field(
        default=1
    )
    weight_quant_method: Optional[str] = field(
        default='bwn'
    )
    weight_featurewise: Optional[bool] = field(
        default=False
    )
    bmt: Optional[bool] = field(
        default=False
    )
    model_parallel_size: Optional[int] = field(
        default=1
    )
    group_norm_size: Optional[int] = field(
        default=1
    )
    quant_ffn_only: Optional[bool] = field(
        default=False
    )
    hadamard_group: Optional[int] = field(
        default=-1,
    )
    blockwise_quant: Optional[bool] = field(
        default=False
    )
    resume_from_fp16: Optional[bool] = field(
        default=False
    )
    smoothquant: Optional[bool] = field(
        default=False
    )
    smoothquant_alpha: Optional[float] = field(
        default=0.5
    )
    binary_attn: Optional[bool] = field(
        default=False,
    )
    weight_blocksize: Optional[str] = field(
        default="-1,-1",
    )
    grad_act: Optional[bool] = field(
        default=False,
    )
    weight_blockscale: Optional[str] = field(
        default='none',
    )
    smoothquant_bitnet: Optional[bool] = field(
        default=False
    )
    input_bits_post: Optional[int] = field(
        default=8
    )
    cal_input_stat: Optional[str] = field(
        default='none'
    )
    rotary_embed: Optional[bool] = field(
        default=False
    )
    no_bias: Optional[bool] = field(
        default=False
    )
    rms_norm: Optional[bool] = field(
        default=False
    )
    binary_query: Optional[bool] = field(
        default=False,
    )
    binary_key: Optional[bool] = field(
        default=False,
    )
    moe_second_expert_threshold: Optional[float] = field(
        default=0.0,
    )
    moe_second_expert_threshold_warmup: Optional[int] = field(
        default=0,
    )
    moe_second_expert_threshold_init: Optional[float] = field(
        default=1e-07,
    )
    key_bits: Optional[int] = field(
        default=1,
    )
    key_quant_method: Optional[str] = field(
        default="bwn",
    )
    moe_expert_noise_threshold_warmup: Optional[int] = field(
        default=0,
    )
    moe_expert_noise_threshold_init: Optional[float] = field(
        default=1e-07,
    )
    moe_expert_noise_threshold: Optional[float] = field(
        default=1e-07,
    )
    moe_ffn_dim: Optional[int] = field(
        default=-1,
    )
    n_kv_heads: Optional[int] = field(
        default=-1,
    )
    # pretrained_dense_ckpt_path: Optional[str] = field(
    #     default="",
    # )
    binary_routing: Optional[bool] = field(
        default=False,
    )
    key_norm: Optional[bool] = field(
        default=False,
    )
    ffn_bits: Optional[int] = field(
        default=-1,
    )
    ffn_quant_method: Optional[str] = field(
        default="",
    )
    attn_bits: Optional[int] = field(
        default=32,
    )
    attn_quant_method: Optional[str] = field(
        default="bwn_per_token_clipped",
    )
    fp8: Optional[bool] = field(
        default=False,
    )
    # quip
    quip_sharp: Optional[bool] = field(
        default=False,
    )
    codebook: Optional[str] = field(
        default="E8P12",
    )
    codebook_version: Optional[int] = field(
        default=0,
    )
    codesz: Optional[int] = field(
        default=8,
    )
    no_fused: Optional[bool] = field(
        default=False,
    )
    idx_dtype: Optional[str] = field(
        default="torch.int16"
    )
    lora_rank: Optional[int] = field(
        default=0,
    )
    model_version: Optional[int] = field(
        default=0,
    )
    outlier_channel_split: Optional[bool] = field(
        default=False,
    )
    packsz: Optional[int] = field(
        default=1,
    )
    rescale_WH: Optional[bool] = field(
        default=False,
    )
    binary_kv: Optional[bool] = field(
        default=False,
    )
    absmean_alpha: Optional[float] = field(
        default=1.0,
    )
    fc2_bits: Optional[int] = field(
        default=-1,
    )
    quant_ffn_output: Optional[bool] = field(
        default=False,
    )
    input_absmean_alpha: Optional[float] = field(
        default=1.0,
    )
    fc2_quant_method: Optional[str] = field(
        default="",
    )

    input_resolution: int = II("task.input_resolution")
    image_token_length: int = II("task.image_token_length")
    patch_size: int = field(
        default=16,
        metadata={"help": "patch size"},
    )
    pretrained_text_gpt_ckpt_path: str = field(
        default="",
        metadata={
            "help": "pretrained text gpt ckpt path"
        },
    )


@register_model("mllm", dataclass=MultimodalLanguageConfig)
class MultimodalLanguageModel(FairseqLanguageModel):
    def __init__(self, args, decoder, patch_embed, image_tokenizer):
        self.args = args
        super().__init__(decoder)
        self.patch_embed = patch_embed
        self.image_tokenizer = image_tokenizer

        for p in self.image_tokenizer.parameters():
            p.requires_grad = False
        
        if getattr(args, "pretrained_text_gpt_ckpt_path", "") != "" and os.path.isfile(args.pretrained_text_gpt_ckpt_path):
            logger.info(f"load {args.pretrained_text_gpt_ckpt_path} for initialization")
            state = checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_text_gpt_ckpt_path)
            pretrained_ckpt_vocab_size = state["model"]["decoder.embed_tokens.weight"].size(0)
            if pretrained_ckpt_vocab_size != self.decoder.embed_tokens.weight.size(0):
                print("pretrained ckpt vocab size: {}".format(pretrained_ckpt_vocab_size))
                for key in ["decoder.embed_tokens.weight", "decoder.output_projection.weight"]:
                    temp_tensor = self.state_dict()[key].clone()
                    temp_tensor[:pretrained_ckpt_vocab_size, :] = state["model"][key]
                    state["model"][key] = temp_tensor

            pos_embed_key = "decoder.embed_positions.weight"
            cur_pos_embed_size = self.state_dict()[pos_embed_key].size(0)
            if state["model"][pos_embed_key].size(0) > cur_pos_embed_size:
                state["model"][pos_embed_key] = state["model"][pos_embed_key][:cur_pos_embed_size]

            loading_result = self.load_state_dict(state["model"], strict=False, args=args)
            print("Missing keys:", loading_result.missing_keys)
            print("Unexpected keys:", loading_result.unexpected_keys)

    @classmethod
    def build_model(cls, args, task):

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_embed_dim
        )

        # patch_embed = VisionEmbedding(
        #     img_size=args.input_resolution, patch_size=args.patch_size, embed_dim=args.decoder_embed_dim,
        # )
        # assert patch_embed.num_position_embeddings() == args.image_token_length
        
        patch_embed = None

        embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                args.decoder_embed_dim,
                task.dictionary.pad(),
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                embed_tokens.weight.shape[1],
                embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.decoder_embed_dim, len(task.dictionary), bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
            )

        if getattr(args, "moe_freq", 0) > 0 and (
            getattr(args, "fp16", False)
            and not getattr(args, "memory_efficient_fp16", False)
            and getattr(args, "ddp_backend", None) != "fully_sharded"
        ):
            assert (
                args.fp16_no_flatten_grads
            ), "If training moe models, set --fp16-no-flatten-grads to calculate correct gradnorm"

        args.ddp_rank = distributed_utils.get_data_parallel_rank()

        config = DecoderConfig()
        config.override(args)

        decoder = LMDecoder(
            config,
            embed_tokens,
            embed_positions,
            output_projection,
            is_encoder_decoder=False,
            dictionary=task.dictionary,
        )

        # pretrained_weights = "/docker/data_dir/conversationhub/wenwan/k3/beitv2/vqkd_encoder_base_decoder_1x768x12_clip-d93179da.pth"
        pretrained_weights = "/home/yulugan/msranlpintern/yulu/code/trm-b3/vqkd_encoder_base_decoder_1x768x12_clip-d93179da.pth"
        image_tokenizer = create_model(
            "vqkd_encoder_base_decoder_1x768x12_clip",
            pretrained=True,
            pretrained_weight=pretrained_weights,
            as_tokenzer=True,
        ).eval()

        return cls(args, decoder, patch_embed, image_tokenizer)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return Embedding(len(dictionary), embed_dim, dictionary.pad())
    
    def forward(self, src_tokens, 
                image_tensors=None, 
                image_tensors_for_tokenizer=None,
                img_gpt_input_mask=None,
                gpt_loss_mask=None, **kwargs):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, seq_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        ##################### looooogs ########################################################################################
        # print("src_tokens shape:", src_tokens.shape) # torch.Size([16, 512])
        # print("image_tensors_for_tokenizer shape:", image_tensors_for_tokenizer.shape) torch.Size([9, 3, 224, 224])
        # if image_tensors is not None:
        #     print("image_tensors shape:", image_tensors.shape) # torch.Size([10, 3, 224, 224])
        #######################################################################################################################
        if image_tensors is not None:
            
            patch_embeddings = image_tensors.reshape(-1, image_tensors.size(-1)) # torch.Size([32768, 770]) ocrloader
            # patch_embeddings = self.patch_embed(image_tensors) # image_tensors: (10, 3, 224, 224), patch_embeddings: (10, 196 ~ (14patches*14patches), 768)
            # patch_embeddings = patch_embeddings.reshape(-1, patch_embeddings.size(-1)) # patch_embeddings: (1960, 768)
            # with torch.no_grad():
            #     image_codes = self.image_tokenizer.get_codebook_indices(ocr_images).reshape(-1)
            #     print("image_codes:",image_codes.shape) # torch.Size([3136])
            #     print("patch_embeddings:",patch_embeddings.shape) # torch.Size([3136, 768])
            image_codes = None
        else:
            patch_embeddings = None
            image_codes = None
        print("image_tensors:ssss", image_tensors)
        ##################### looooogs ########################################################################################
        # print("patch_embeddings shape:", patch_embeddings.shape) # torch.Size([1960, 768])
        # print("image_codes shape:", image_codes.shape) # torch.Size([1960])
        #######################################################################################################################
        
        self_attn_padding_mask = src_tokens.eq(self.decoder.dictionary.pad())
        # self.decoder <- LMDecoder
        # patch_embeddings: (3136,768) ocrloader
        
        return self.decoder(src_tokens, self_attn_padding_mask, 
                            patch_embeddings=patch_embeddings, #  patch_embeddings: torch.Size([16*4096, 770]) ocrloader
                            img_gpt_input_mask=img_gpt_input_mask,
                            **kwargs), image_codes


class LMDecoder(Decoder, FairseqIncrementalDecoder):
    def max_positions(self):
        return self.args.max_target_positions
        # return self.embed_positions.max_positions

    def reorder_incremental_state_scripting(
        self,
        incremental_state,
        new_order,
    ):
        for module in incremental_state:
            for key in incremental_state[module]:
                result = incremental_state[module][key].index_select(0, new_order)
                incremental_state[module][key] = result
    
    def forward_embedding(
        self,
        tokens,
        token_embedding=None,
        incremental_state=None,
        patch_embeddings = None,
        img_gpt_input_mask = None,
        infer_first_step = False,
    ):
        '''
        tokens是full token，应该包含文本token和图像token
        token_embedding 是 tokens过nn.embedding查词表后得到的embedding
        然后token_embedding中图像部分之前被占位符替换，会被patch embedding重新赋值
        Return: 
            -- x: token_embedding * scale (+position)
        '''
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                tokens, incremental_state=incremental_state
            )

        if incremental_state is not None and not infer_first_step:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens) # token_embedding: torch.Size([16, 512, 768])

        gpt_embed_output = token_embedding
        if patch_embeddings is not None:
            # print("img_gpt_input_mask:", img_gpt_input_mask.shape) #torch.Size([16, 512]) torch.Size([2, 2052])
            # print("gpt_embed_output:", gpt_embed_output.shape) #torch.Size([16, 512, 768]) torch.Size([2, 2052, 768])
            # print("patch_embeddings:", patch_embeddings.shape) #torch.Size([32768, 768]) torch.Size([4096 (2*2052), 768])
            gpt_embed_output[img_gpt_input_mask] = patch_embeddings
        x = embed = self.embed_scale * gpt_embed_output
        print("patch_embeddings:ssss", patch_embeddings)
        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def forward(
        self,
        prev_output_tokens,
        self_attn_padding_mask=None,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        return_all_hiddens=False,
        token_embeddings=None,
        infer_first_step=False,
        **kwargs
    ):
        '''
        -- prev_output_tokens: full tokens
        
        '''
        # embed tokens and positions, token_embeddings is none here; x is full tokens (text + img tokens, def see in gpt_ocr_loader.py:full_tokens(def _preprare))
        x, _ = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state, infer_first_step=infer_first_step, **kwargs
        )
        # relative position
        self_attn_rel_pos_bias = None
        slen = prev_output_tokens.size(1)
        if self.self_attn_relative_position is not None:
            self_attn_rel_pos_bias = self.self_attn_relative_position(
                batch_size=x.size(0), qlen=slen, klen=slen
            )
            if incremental_state is not None:
                self_attn_rel_pos_bias = self_attn_rel_pos_bias[-1:, :, :]
        cross_attn_rel_pos_bias = None
        if self.cross_attn_relative_position is not None:
            cross_attn_rel_pos_bias = self.cross_attn_relative_position(
                batch_size=x.size(0),
                qlen=slen,
                klen=encoder_out["encoder_out"].size(1),
            )
            if incremental_state is not None:
                cross_attn_rel_pos_bias = cross_attn_rel_pos_bias[-1:, :, :]
        # decoder layers
        inner_states = [x]

        if encoder_out is None:
            l_aux = []
        else:
            l_aux = encoder_out["l_aux"] if "l_aux" in encoder_out else []

        # generate position_ids for rotary embedding, 
        # follow transformers.models.llama.modeling_llama.prepare_inputs_for_generation
        seq_length = x.size(1)
        past_key_values_length = 0
        if incremental_state is not None:
            if len(incremental_state) > 0:
                past_key_values_length = incremental_state[0]["prev_key"].shape[2]
            else:
                past_key_values_length = 0

        device = prev_output_tokens.device 
        # Usually, to be consistent with Fairseq, we have to start from 2 in position embeddings
        # but this `position_ids` is for rotary embedding which needs to starts from 0, so we do NOT +2 here.
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        for idx, layer in enumerate(self.layers):
            if incremental_state is None or infer_first_step:
                self_attn_mask = torch.triu(
                    torch.zeros([x.size(1), x.size(1)])
                    .float()
                    .fill_(float("-inf"))
                    .type_as(x),
                    1,
                )
                if infer_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                self_attn_mask = None
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x, layer_attn, _, l_aux_i = layer(
                x,
                encoder_out["encoder_out"] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"]
                if encoder_out is not None
                else None,
                incremental_state[idx] if incremental_state is not None else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                self_attn_rel_pos=self_attn_rel_pos_bias,
                cross_attn_rel_pos=cross_attn_rel_pos_bias,
                position_ids=position_ids # pass position_ids for rotary embedding
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only:
            x = self.output_layer(x)
            
        return x, {
            "inner_states": inner_states,
            "l_aux": l_aux,
            "attn": None,
        }


class VisionEmbedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        prepend_cls_token=False,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if prepend_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def num_position_embeddings(self):
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        batch_size, seq_len, _ = x.size()

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                batch_size, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed

        return x


@register_model_architecture("mllm", "mllm_base")
def base_mllm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.n_kv_heads = getattr(args, "n_kv_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.base_layers = getattr(args, "base_layers", 0)
    args.base_sublayers = getattr(args, "base_sublayers", 1)
    args.base_shuffle = getattr(args, "base_shuffle", False)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True

@register_model_architecture("mllm", "mllm_small")
def small_mllm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.n_kv_heads = getattr(args, "n_kv_heads", 12)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True

@register_model_architecture("mllm", "mllm_small_v2")
def small_mllm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.n_kv_heads = getattr(args, "n_kv_heads", 12)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "silu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True

@register_model_architecture("mllm", "mllm_large")
def large_mllm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.n_kv_heads = getattr(args, "n_kv_heads", 16)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)
    args.flash_attention = getattr(args, "flash_attention", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True

@register_model_architecture("mllm", "mllm_large_v2")
def large_mllm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2736)
    args.decoder_layers = getattr(args, "decoder_layers", 20)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.n_kv_heads = getattr(args, "n_kv_heads", 16)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "silu")
    args.flash_attention = getattr(args, "flash_attention", True)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True

@register_model_architecture("mllm", "mllm_xl")
def xl_mllm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.n_kv_heads = getattr(args, "n_kv_heads", 16)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True

@register_model_architecture("mllm", "mllm_xl_v2")
def xl_mllm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 5504)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.n_kv_heads = getattr(args, "n_kv_heads", 32)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "silu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("mllm", "mllm_xl_v3")
def xl_mllm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 5460)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.n_kv_heads = getattr(args, "n_kv_heads", 32)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "silu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("mllm", "mllm_6b7")
def mllm_6b7_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 4096)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 16384)
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.n_kv_heads = getattr(args, "n_kv_heads", 32)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True

@register_model_architecture("mllm", "mllm_3b")
def mllm_3b_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 10240)
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.n_kv_heads = getattr(args, "n_kv_heads", 32)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
