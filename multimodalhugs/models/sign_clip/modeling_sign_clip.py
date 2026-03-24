import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.utils.registry import register_model

logger = logging.getLogger(__name__)


@dataclass
class SignCLIPOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_sign: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    sign_embeds: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    sign_hidden_states: Optional[torch.FloatTensor] = None
    text_hidden_states: Optional[torch.FloatTensor] = None


class SignCLIPProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, l2_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.l2_norm = l2_norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(hidden_states)
        if self.l2_norm:
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)
        return hidden_states


class SignCLIPVideoConv1D(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, kernel_size: int = 17):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    input_dim,
                    input_dim,
                    kernel_size=kernel_size,
                    padding="same",
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.swapaxes(hidden_states, 1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return torch.swapaxes(hidden_states, 1, 2)


class SignCLIPVideoTokenMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dropout: float, activation: str):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = getattr(F, activation) if hasattr(F, activation) else F.gelu
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        return hidden_states


@register_model("sign_clip")
class SignCLIPModel(PreTrainedModel):
    """
    Hugging Face native SignCLIP model.

    This implementation migrates the core model behavior from the fairseq/MMPT
    SignCLIP setup into a `PreTrainedModel`:
    - sign tower with projected frame tokens
    - text tower with a Hugging Face text encoder
    - masked mean pooling
    - optional projection heads
    - CLIP-style contrastive logits and loss
    """

    config_class = SignCLIPConfig
    base_model_prefix = "sign_clip"
    is_parallelizable = True
    _keep_in_fp32_modules = []
    _no_split_modules = []

    def __init__(self, config: SignCLIPConfig):
        super().__init__(config)

        self.sign_encoder = self._build_sign_encoder(config)
        self.text_encoder = self._build_text_encoder(config)
        self.sign_conv1d = self._build_sign_conv1d(config)
        self.sign_token_mlp = self._build_sign_token_mlp(config)
        self.sign_projection = self._build_projection(config, self.sign_encoder.config.hidden_size)
        self.text_projection = self._build_projection(config, self.text_encoder.config.hidden_size)
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(config.logit_scale_init_value, dtype=torch.float32)
        )
        self._set_trainable_parameters(config)
        self.post_init()

    def _build_encoder_config(
        self,
        encoder_type: str,
        encoder_config: Optional[dict],
        pretrained_name: Optional[str],
        *,
        sign_side: bool,
    ):
        if encoder_config is not None:
            cfg = AutoConfig.for_model(encoder_type, **encoder_config)
        elif pretrained_name is not None:
            cfg = AutoConfig.from_pretrained(pretrained_name)
        else:
            cfg = AutoConfig.for_model(encoder_type)

        if sign_side:
            if self.config.sign_hidden_size is not None:
                cfg.hidden_size = self.config.sign_hidden_size
            if self.config.num_hidden_sign_encoder_layers is not None:
                if hasattr(cfg, "num_hidden_layers"):
                    cfg.num_hidden_layers = self.config.num_hidden_sign_encoder_layers
                elif hasattr(cfg, "num_layers"):
                    cfg.num_layers = self.config.num_hidden_sign_encoder_layers
            if hasattr(cfg, "max_position_embeddings"):
                cfg.max_position_embeddings = self.config.sign_max_position_embeddings

        if hasattr(cfg, "hidden_dropout_prob"):
            cfg.hidden_dropout_prob = self.config.hidden_dropout_prob
        if hasattr(cfg, "attention_probs_dropout_prob"):
            cfg.attention_probs_dropout_prob = self.config.attention_probs_dropout_prob
        return cfg

    def _build_sign_encoder(self, config: SignCLIPConfig):
        sign_config = self._build_encoder_config(
            config.sign_encoder_type,
            config.sign_encoder_config,
            config.pretrained_sign_encoder,
            sign_side=True,
        )
        if config.pretrained_sign_encoder is not None:
            return AutoModel.from_pretrained(config.pretrained_sign_encoder, config=sign_config)
        return AutoModel.from_config(sign_config)

    def _build_text_encoder(self, config: SignCLIPConfig):
        text_config = self._build_encoder_config(
            config.text_encoder_type,
            config.text_encoder_config,
            config.pretrained_text_encoder,
            sign_side=False,
        )
        if config.pretrained_text_encoder is not None:
            return AutoModel.from_pretrained(config.pretrained_text_encoder, config=text_config)
        return AutoModel.from_config(text_config)

    def _build_sign_conv1d(self, config: SignCLIPConfig):
        if config.sign_conv1d_layers and config.sign_conv1d_layers > 0:
            return SignCLIPVideoConv1D(
                input_dim=config.sign_input_dim,
                num_layers=config.sign_conv1d_layers,
            )
        return None

    def _build_sign_token_mlp(self, config: SignCLIPConfig):
        hidden_size = self.sign_encoder.config.hidden_size
        activation = getattr(self.sign_encoder.config, "hidden_act", "gelu")
        return SignCLIPVideoTokenMLP(
            input_dim=config.sign_input_dim,
            hidden_size=hidden_size,
            dropout=config.hidden_dropout_prob,
            activation=activation,
        )

    def _build_projection(self, config: SignCLIPConfig, input_dim: int):
        if not config.use_projection:
            return None
        return SignCLIPProjection(
            in_dim=input_dim,
            out_dim=config.projection_dim,
            l2_norm=config.projection_l2_norm,
        )

    def _set_trainable_parameters(self, config: SignCLIPConfig):
        self._set_module_requires_grad(self.sign_encoder, not config.freeze_sign_encoder)
        self._set_module_requires_grad(self.text_encoder, not config.freeze_text_encoder)
        if self.sign_projection is not None:
            self._set_module_requires_grad(self.sign_projection, not config.freeze_sign_projection)
        if self.text_projection is not None:
            self._set_module_requires_grad(self.text_projection, not config.freeze_text_projection)

    @staticmethod
    def _set_module_requires_grad(module: nn.Module, requires_grad: bool):
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _build_sign_attention_mask(
        self,
        sign_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sign_attention_mask.size(0)
        cls_mask = torch.zeros((batch_size, 1), dtype=sign_attention_mask.dtype, device=sign_attention_mask.device)
        sep_mask = torch.ones((batch_size, 1), dtype=sign_attention_mask.dtype, device=sign_attention_mask.device)
        return torch.cat([cls_mask, sign_attention_mask, sep_mask], dim=1)

    def _build_sign_inputs_embeds(
        self,
        sign_inputs: torch.Tensor,
        sign_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected_tokens = sign_inputs
        if self.sign_conv1d is not None:
            projected_tokens = self.sign_conv1d(projected_tokens)
        projected_tokens = self.sign_token_mlp(projected_tokens)

        batch_size = projected_tokens.size(0)
        input_embeddings = self.sign_encoder.get_input_embeddings()

        cls_token_id = getattr(self.sign_encoder.config, "cls_token_id", 101)
        sep_token_id = getattr(self.sign_encoder.config, "sep_token_id", 102)
        cls_tokens = torch.full(
            (batch_size, 1),
            cls_token_id,
            dtype=torch.long,
            device=projected_tokens.device,
        )
        sep_tokens = torch.full(
            (batch_size, 1),
            sep_token_id,
            dtype=torch.long,
            device=projected_tokens.device,
        )

        cls_embeds = input_embeddings(cls_tokens)
        sep_embeds = input_embeddings(sep_tokens)
        return torch.cat([cls_embeds, projected_tokens, sep_embeds], dim=1)

    @staticmethod
    def _masked_mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.to(hidden_states.dtype)
        mask = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.bmm(hidden_states.transpose(2, 1), mask.unsqueeze(2)).squeeze(-1)

    def get_sign_features(
        self,
        sign_inputs: torch.FloatTensor,
        sign_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        if sign_attention_mask is None:
            sign_attention_mask = torch.ones(
                sign_inputs.shape[:2],
                dtype=torch.long,
                device=sign_inputs.device,
            )

        inputs_embeds = self._build_sign_inputs_embeds(sign_inputs, sign_attention_mask)
        attention_mask = self._build_sign_attention_mask(sign_attention_mask)
        token_type_ids = torch.zeros_like(attention_mask, dtype=torch.long)

        outputs = self.sign_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states,
            return_dict=True,
        )
        sign_hidden_states = outputs.last_hidden_state
        sign_pool_mask = self._build_sign_attention_mask(sign_attention_mask)
        sign_features = self._masked_mean_pool(sign_hidden_states, sign_pool_mask)
        if self.sign_projection is not None:
            sign_features = self.sign_projection(sign_features)
        return sign_features, sign_hidden_states

    def get_text_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states,
            return_dict=True,
        )
        text_hidden_states = outputs.last_hidden_state
        text_pool_mask = attention_mask.clone()
        text_pool_mask[:, 0] = 0
        text_features = self._masked_mean_pool(text_hidden_states, text_pool_mask)
        if self.text_projection is not None:
            text_features = self.text_projection(text_features)
        return text_features, text_hidden_states

    def _compute_logits(
        self,
        sign_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        if not self.config.use_projection and self.config.projection_l2_norm:
            sign_features = F.normalize(sign_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=self.config.max_logit_scale)
        logits_per_sign = logit_scale * torch.matmul(sign_features, text_features.transpose(0, 1))
        logits_per_text = logits_per_sign.transpose(0, 1)
        return logits_per_sign, logits_per_text

    @staticmethod
    def _compute_contrastive_loss(
        logits_per_sign: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> torch.Tensor:
        labels = torch.arange(logits_per_sign.size(0), device=logits_per_sign.device)
        sign_loss = F.cross_entropy(logits_per_sign, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        return 0.5 * (sign_loss + text_loss)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        sign_inputs: Optional[torch.FloatTensor] = None,
        sign_attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if sign_inputs is None:
            raise ValueError("`sign_inputs` must be provided for SignCLIPModel.forward().")
        if input_ids is None:
            raise ValueError("`input_ids` must be provided for SignCLIPModel.forward().")

        sign_embeds, sign_hidden_states = self.get_sign_features(
            sign_inputs=sign_inputs,
            sign_attention_mask=sign_attention_mask,
        )
        text_embeds, text_hidden_states = self.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        logits_per_sign, logits_per_text = self._compute_logits(sign_embeds, text_embeds)
        loss = self._compute_contrastive_loss(logits_per_sign, logits_per_text) if return_loss else None

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if not return_dict:
            return (
                loss,
                logits_per_sign,
                logits_per_text,
                sign_embeds,
                text_embeds,
                sign_hidden_states,
                text_hidden_states,
            )

        return SignCLIPOutput(
            loss=loss,
            logits_per_sign=logits_per_sign,
            logits_per_text=logits_per_text,
            sign_embeds=sign_embeds,
            text_embeds=text_embeds,
            sign_hidden_states=sign_hidden_states,
            text_hidden_states=text_hidden_states,
        )
