import math

import torch

from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.models.sign_clip.modeling_sign_clip import SignCLIPModel
from multimodalhugs.tasks.contrastive.config_classes import (
    ContrastiveDataArguments,
    ContrastiveModelArguments,
)
from multimodalhugs.tasks.contrastive.contrastive_training import _load_config, _load_model


def test_runtime_override_disables_distributed_negatives(tmp_path):
    SignCLIPConfig(use_distributed_negatives=True).save_pretrained(tmp_path)

    config = _load_config(
        ContrastiveModelArguments(
            config_name=str(tmp_path),
            use_distributed_negatives=False,
        )
    )

    assert config.use_distributed_negatives is False


def test_eval_split_defaults_to_validation_and_accepts_test():
    assert ContrastiveDataArguments().eval_split_name == "validation"
    assert ContrastiveDataArguments(eval_split_name="test").eval_split_name == "test"


def test_runtime_override_sets_siglip_loss(tmp_path):
    SignCLIPConfig(contrastive_loss_type="clip").save_pretrained(tmp_path)

    config = _load_config(
        ContrastiveModelArguments(
            config_name=str(tmp_path),
            contrastive_loss_type="siglip",
            logit_bias_init_value=-10.0,
            logit_scale_init_value=math.log(10.0),
            siglip_distributed_implementation="ring",
        )
    )

    assert config.contrastive_loss_type == "siglip"
    assert config.logit_bias_init_value == -10.0
    assert config.logit_scale_init_value == math.log(10.0)
    assert config.siglip_distributed_implementation == "ring"


def test_runtime_override_resets_loaded_logit_parameters(tmp_path):
    encoder_config = {
        "vocab_size": 32,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "max_position_embeddings": 16,
        "type_vocab_size": 2,
    }
    config = SignCLIPConfig(
        sign_encoder_type="bert",
        sign_encoder_config=encoder_config,
        text_encoder_type="bert",
        text_encoder_config=encoder_config,
        sign_input_dim=4,
        projection_dim=8,
        logit_scale_init_value=4.0,
        logit_bias_init_value=3.0,
    )
    SignCLIPModel(config).save_pretrained(tmp_path)

    model_args = ContrastiveModelArguments(
        model_name_or_path=str(tmp_path),
        logit_scale_init_value=math.log(10.0),
        logit_bias_init_value=-10.0,
    )
    loaded_config = _load_config(model_args)
    loaded_model = _load_model(model_args, loaded_config)

    torch.testing.assert_close(loaded_model.logit_scale, torch.tensor(math.log(10.0)))
    torch.testing.assert_close(loaded_model.logit_bias, torch.tensor(-10.0))
