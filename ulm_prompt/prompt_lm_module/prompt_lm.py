import json
import math
from ast import arg
from dataclasses import dataclass, field
from lib2to3.pgen2 import token
from typing import Any, Dict, List, Optional

import torch
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (DEFAULT_MIN_PARAMS_TO_WRAP, Embedding,
                                        TransformerConfig,
                                        TransformerDecoderBase)
from fairseq.models.transformer_lm import (TransformerLanguageModel,
                                           TransformerLanguageModelConfig,
                                           transformer_lm_big)
from fairseq.modules import SinusoidalPositionalEmbedding
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq_usr.transformer_decoder import TransformerDecoderPromptBase
from torch import Tensor
from torch.nn import Dropout, ModuleList

DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class TransformerLanguageModelPromptConfig(TransformerLanguageModelConfig):
    prefix_prompt_length: int = field(default=10, metadata={"help": "prompt length"})
    infix_prompt_length: int = field(default=10, metadata={"help": "infix prompt length"})
    deep_prompt: bool = field(default=True, metadata={"help": "if using deep prompt"})
    use_sep_token: bool = field(default=True, metadata={"help": "if using sep token <s> "})
    fine_tune: bool = field(default=False, metadata={"help": "if finetuning the whole model"})


class TransformerDecoderPrompt(TransformerDecoderPromptBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        embed_sep_token,
        prefix_embed_prompts,
        infix_embed_prompts,
        deep_prompt,
        deep_key_embed_prompts,
        deep_value_embed_prompts,
        prefix_prompt_length,
        infix_prompt_length,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn,
            output_projection,
        )
        self.use_sep_token = args.use_sep_token
        self.sep = 0
        self.prefix_embed_prompts = prefix_embed_prompts
        self.infix_embed_prompts = infix_embed_prompts
        self.sep_embed = embed_sep_token
        self.deep_prompt = deep_prompt
        self.deep_key_embed_prompts = deep_key_embed_prompts
        self.deep_value_embed_prompts = deep_value_embed_prompts
        self.prefix_prompt_length = prefix_prompt_length
        self.infix_prompt_length = infix_prompt_length
        self.prompt_dropout = Dropout(p=0.1)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
        )

        # x.shape : [32, 1, 1024]
        if not features_only:
            x = self.output_layer(x)
        # x.shape : [32, 1. 104]
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            src_lengths,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # === not using incremental decoding === #
        incremental_state = None
        # ====================================== #
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert enc.size()[1] == bs, f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # ===== <embed positions> ===== #
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state)

        token_embed = self.embed_tokens(prev_output_tokens)

        # ======= <sep> ======= #
        sep_embed = self.sep_embed(torch.tensor(self.sep).cuda())
        # replace sep with trainable sep token
        if self.use_sep_token:
            sep_position = (prev_output_tokens == 0).nonzero()
            for (i, p) in sep_position:
                token_embed[int(i)][int(p)] = sep_embed
        # ======= </sep> ======= #
        x = self.embed_scale * token_embed

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        # ===== </embed positions> ===== #

        # ===== <!!! Prompting !!!> =====#
        # concatenation: [p1, p2, p3, ..., pN] + [original tokens]
        # No positional embedding on prompts

        if self.prefix_prompt_length > 0:
            # prefix prompt
            prefix_task_embed_idx = torch.tensor(list(range(self.prefix_prompt_length))).cuda()
            prefix_task_embed = self.prefix_embed_prompts(prefix_task_embed_idx)
            prefix_task_embed = self.prompt_dropout(prefix_task_embed)

        if self.infix_prompt_length > 0:
            # infix prompt
            infix_task_embed_idx = torch.tensor(list(range(self.infix_prompt_length))).cuda()
            infix_task_embed = self.infix_embed_prompts(infix_task_embed_idx)
            infix_task_embed = self.prompt_dropout(infix_task_embed)

        # ===== <concat prompt> ===== #
        x_sep = None
        prompt_type = None
        # condition 1: prefix prompt and infix prompt at input
        if self.prefix_prompt_length > 0 and self.infix_prompt_length > 0:
            x_sep = (prev_output_tokens == 0).nonzero()
            x = torch.concat(
                [
                    torch.concat(
                        (prefix_task_embed, src[:sep_i], infix_task_embed, src[sep_i:]),
                        dim=0,
                    ).unsqueeze(0)
                    for src, (i, sep_i) in zip(x, x_sep)
                ]
            )
            prompt_type = "prefix_infix"

        # condition 2: prefix prompt at input
        elif self.prefix_prompt_length > 0 and self.infix_prompt_length == 0:
            x = torch.concat([torch.concat((prefix_task_embed, src), dim=0).unsqueeze(0) for src in x])
            prompt_type = "prefix"

        # condition 3: infix prompt at input
        elif self.prefix_prompt_length == 0 and self.infix_prompt_length > 0:
            x_sep = (prev_output_tokens == 0).nonzero()
            x = torch.concat(
                [
                    torch.concat(
                        (src[:sep_i], infix_task_embed, src[sep_i:]),
                        dim=0,
                    ).unsqueeze(0)
                    for src, (i, sep_i) in zip(x, x_sep)
                ]
            )
            prompt_type = "infix"

        # ===== </concat prompt> ===== #
        # dummy padding for attention padding mask
        pad = (
            torch.tensor(self.dictionary.bos_index)
            .unsqueeze(-1)
            .repeat(
                prev_output_tokens.size(0),
                self.prefix_prompt_length + self.infix_prompt_length,
            )
            .to(prev_output_tokens.device)
        )

        prev_output_tokens = torch.concat((pad, prev_output_tokens), dim=1)
        # ===== </!!! Prompting !!!> =====#

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # === <!!! deep prompt tuning!!!> === #
            if self.deep_prompt and prompt_type == "prefix":
                deep_key_prompt = self.deep_key_embed_prompts[idx](prefix_task_embed_idx)
                deep_value_prompt = self.deep_value_embed_prompts[idx](prefix_task_embed_idx)
                deep_key_prompt = self.prompt_dropout(deep_key_prompt)
                deep_value_prompt = self.prompt_dropout(deep_value_prompt)
            elif self.deep_prompt and prompt_type == "infix":
                deep_key_prompt = self.deep_key_embed_prompts[idx](infix_task_embed_idx)
                deep_value_prompt = self.deep_value_embed_prompts[idx](infix_task_embed_idx)
                deep_key_prompt = self.prompt_dropout(deep_key_prompt)
                deep_value_prompt = self.prompt_dropout(deep_value_prompt)
            else:
                deep_key_prompt = None
                deep_value_prompt = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                key_prompt=deep_key_prompt,
                value_prompt=deep_value_prompt,
                prompt_type=prompt_type,
                x_sep=x_sep,
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
            # === </!!! deep prompt tuning!!!> === #

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if src_lengths is not None:
            return x, {
                "attn": [attn],
                "inner_states": inner_states,
                "prompt_length": self.prefix_prompt_length + self.infix_prompt_length,
            }
        else:
            return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to  vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]


@register_model("transformer_lm_prompt", dataclass=TransformerLanguageModelPromptConfig)
class TransformerLanguageModelPrompt(TransformerLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):

        if safe_getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = safe_getattr(args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS)

        embed_tokens = cls.build_embedding(args, task.source_dictionary, args.decoder_input_dim)
        sep_embed_token = cls.build_sep_embedding(args, args.decoder_input_dim)

        # === <prompt> === #
        deep_prompt = args.deep_prompt
        prefix_prompt_length = args.prefix_prompt_length
        infix_prompt_length = args.infix_prompt_length

        prefix_embed_prompts = cls.build_prefix_prompt_embedding(args, args.decoder_input_dim)
        infix_embed_prompts = cls.build_infix_prompt_embedding(args, args.decoder_input_dim)

        if deep_prompt and not args.fine_tune:
            (
                deep_key_embed_prompts,
                deep_value_embed_prompts,
            ) = cls.build_deep_prompt_embedding(args, args.decoder_input_dim)
        else:
            deep_key_embed_prompts = None
            deep_value_embed_prompts = None

        # === </prompt> === #
        decoder = TransformerDecoderPrompt(
            args,
            task.target_dictionary,
            embed_tokens,
            sep_embed_token,
            prefix_embed_prompts,
            infix_embed_prompts,
            deep_prompt,
            deep_key_embed_prompts,
            deep_value_embed_prompts,
            prefix_prompt_length,
            infix_prompt_length,
            no_encoder_attn=True,
        )

        if not args.fine_tune:
            # ===== <!!! Prompting !!!> =====#
            # fix the pretrained model parameters
            # make prompt parameters trainable
            # make sure these trainable parameters will (only) be saved when saving prompt checkpoint
            for name, p in decoder.named_parameters():
                p.requires_grad = False
                if "embed_prompts" in name:
                    p.requires_grad = True
                if "sep_embed" in name:
                    p.requires_grad = True

        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        """
        e.g.
            len(dictionary): 104
            embed_dim: 1024
            dictionary.pad(): 1
        """
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens

    @classmethod
    def build_prefix_prompt_embedding(cls, args, embed_dim, path=None):
        embed_prompts = torch.nn.Embedding(args.prefix_prompt_length, embed_dim, padding_idx=None)
        return embed_prompts

    @classmethod
    def build_infix_prompt_embedding(cls, args, embed_dim, path=None):
        embed_prompts = torch.nn.Embedding(args.infix_prompt_length, embed_dim, padding_idx=None)
        return embed_prompts

    @classmethod
    def build_deep_prompt_embedding(cls, args, embed_dim, path=None):
        num_layers = args.decoder_layers
        if args.prefix_prompt_length > 0 and args.infix_prompt_length == 0:
            key_embed_prompts = ModuleList(
                [torch.nn.Embedding(args.prefix_prompt_length, embed_dim, padding_idx=None) for i in range(num_layers)]
            )
            value_embed_prompts = ModuleList(
                [torch.nn.Embedding(args.prefix_prompt_length, embed_dim, padding_idx=None) for i in range(num_layers)]
            )
        elif args.prefix_prompt_length == 0 and args.infix_prompt_length > 0:
            key_embed_prompts = ModuleList(
                [torch.nn.Embedding(args.infix_prompt_length, embed_dim, padding_idx=None) for i in range(num_layers)]
            )
            value_embed_prompts = ModuleList(
                [torch.nn.Embedding(args.infix_prompt_length, embed_dim, padding_idx=None) for i in range(num_layers)]
            )
        elif args.prefix_prompt_length == 0 and args.infix_prompt_length == 0:
            return None, None
        else:
            raise NotImplementedError
        return key_embed_prompts, value_embed_prompts

    @classmethod
    def build_sep_embedding(cls, args, embed_dim, path=None):
        sep_embed = torch.nn.Embedding(1, embed_dim, padding_idx=None)
        return sep_embed


@register_model_architecture("transformer_lm_prompt", "transformer_lm_big_prompt")
def transformer_lm_big_prompt(args):
    """
    Prompting arguments
    * prefix_prompt_length
    * infix_prompt_length
    * use_sep_token
    """
    args.prefix_prompt_length = safe_getattr(args, "prefix_prompt_length", 10)
    args.infix_prompt_length = safe_getattr(args, "infix_prompt_length", 10)
    args.deep_prompt = safe_getattr(args, "deep_prompt", True)
    args.use_sep_token = safe_getattr(args, "use_sep_token", True)
    args.fine_tune = safe_getattr(args, "fine_tune", False)

    transformer_lm_big(args)
