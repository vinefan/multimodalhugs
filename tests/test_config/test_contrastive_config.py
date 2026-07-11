from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.tasks.contrastive.config_classes import ContrastiveModelArguments
from multimodalhugs.tasks.contrastive.contrastive_training import _load_config


def test_runtime_override_disables_distributed_negatives(tmp_path):
    SignCLIPConfig(use_distributed_negatives=True).save_pretrained(tmp_path)

    config = _load_config(
        ContrastiveModelArguments(
            config_name=str(tmp_path),
            use_distributed_negatives=False,
        )
    )

    assert config.use_distributed_negatives is False


def test_runtime_override_sets_siglip_loss(tmp_path):
    SignCLIPConfig(contrastive_loss_type="clip").save_pretrained(tmp_path)

    config = _load_config(
        ContrastiveModelArguments(
            config_name=str(tmp_path),
            contrastive_loss_type="siglip",
            logit_bias_init_value=-10.0,
        )
    )

    assert config.contrastive_loss_type == "siglip"
    assert config.logit_bias_init_value == -10.0
