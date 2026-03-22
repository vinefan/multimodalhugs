import logging
from typing import Optional

import torch
from transformers import PreTrainedModel

from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.utils.registry import register_model

logger = logging.getLogger(__name__)


@register_model("sign_clip")
class SignCLIPModel(PreTrainedModel):
    """
    Minimal SignCLIP model skeleton.

    The implementation is intentionally deferred; this class exists so we can
    incrementally add sign/text encoders, projection heads, and contrastive
    training utilities without changing the public module layout.
    """

    config_class = SignCLIPConfig
    base_model_prefix = "sign_clip"
    is_parallelizable = True
    _keep_in_fp32_modules = []
    _no_split_modules = []

    def __init__(self, config: SignCLIPConfig):
        super().__init__(config)
        self.sign_encoder = None
        self.text_encoder = None
        self.sign_projection = None
        self.text_projection = None
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(config.logit_scale_init_value, dtype=torch.float32)
        )
        self.post_init()

    def get_sign_features(self, *args, **kwargs):
        raise NotImplementedError("Sign encoder skeleton is not implemented yet.")

    def get_text_features(self, *args, **kwargs):
        raise NotImplementedError("Text encoder skeleton is not implemented yet.")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "SignCLIPModel is currently a scaffold. Implement encoders, projections, "
            "and contrastive loss before calling forward()."
        )
