from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class SignCLIPConfig(PretrainedConfig):
    """
    Configuration skeleton for a SignCLIP-style model.

    This placeholder keeps the API aligned with Hugging Face configs while we
    iterate on the actual dual-encoder design.
    """

    model_type = "sign_clip"

    def __init__(
        self,
        model_type: str = "sign_clip",
        sign_encoder_type: Optional[str] = None,
        sign_encoder_config: Optional[Dict[str, Any]] = None,
        text_encoder_type: Optional[str] = None,
        text_encoder_config: Optional[Dict[str, Any]] = None,
        projection_dim: int = 512,
        logit_scale_init_value: float = 2.6592,
        freeze_sign_encoder: bool = False,
        freeze_text_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_type = model_type
        self.sign_encoder_type = sign_encoder_type
        self.sign_encoder_config = sign_encoder_config
        self.text_encoder_type = text_encoder_type
        self.text_encoder_config = text_encoder_config
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.freeze_sign_encoder = freeze_sign_encoder
        self.freeze_text_encoder = freeze_text_encoder

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
