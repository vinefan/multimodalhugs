from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class SignCLIPConfig(PretrainedConfig):
    """
    Configuration for a SignCLIP-style dual-encoder model.

    This config is designed to support a Hugging Face native migration of the
    fairseq/MMPT SignCLIP implementation. The current focus is the model layer:
    sign tower, text tower, pooling, projection, and contrastive scoring.
    """

    model_type = "sign_clip"

    def __init__(
        self,
        model_type: str = "sign_clip",
        sign_encoder_type: str = "bert",
        sign_encoder_config: Optional[Dict[str, Any]] = None,
        pretrained_sign_encoder: Optional[str] = None,
        text_encoder_type: str = "bert",
        text_encoder_config: Optional[Dict[str, Any]] = None,
        pretrained_text_encoder: Optional[str] = None,
        sign_input_dim: int = 512,
        sign_hidden_size: Optional[int] = None,
        num_hidden_sign_encoder_layers: Optional[int] = None,
        sign_max_position_embeddings: int = 512,
        sign_use_seg_emb: bool = False,
        sign_conv1d_layers: int = 0,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_projection: bool = True,
        projection_dim: int = 512,
        projection_l2_norm: bool = True,
        logit_scale_init_value: float = 2.6592,
        max_logit_scale: float = 100.0,
        freeze_sign_encoder: bool = False,
        freeze_text_encoder: bool = False,
        freeze_sign_projection: bool = False,
        freeze_text_projection: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_type = model_type
        self.sign_encoder_type = sign_encoder_type
        self.sign_encoder_config = sign_encoder_config
        self.pretrained_sign_encoder = pretrained_sign_encoder
        self.text_encoder_type = text_encoder_type
        self.text_encoder_config = text_encoder_config
        self.pretrained_text_encoder = pretrained_text_encoder
        self.sign_input_dim = sign_input_dim
        self.sign_hidden_size = sign_hidden_size
        self.num_hidden_sign_encoder_layers = num_hidden_sign_encoder_layers
        self.sign_max_position_embeddings = sign_max_position_embeddings
        self.sign_use_seg_emb = sign_use_seg_emb
        self.sign_conv1d_layers = sign_conv1d_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.projection_l2_norm = projection_l2_norm
        self.logit_scale_init_value = logit_scale_init_value
        self.max_logit_scale = max_logit_scale
        self.freeze_sign_encoder = freeze_sign_encoder
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_sign_projection = freeze_sign_projection
        self.freeze_text_projection = freeze_text_projection
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
