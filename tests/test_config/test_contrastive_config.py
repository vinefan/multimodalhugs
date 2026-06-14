from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.tasks.contrastive.config_classes import ContrastiveModelArguments
from multimodalhugs.tasks.contrastive.contrastive_training import (
    _build_fixed_train_batch,
    _load_config,
)


def test_runtime_override_disables_distributed_negatives(tmp_path):
    SignCLIPConfig(use_distributed_negatives=True).save_pretrained(tmp_path)

    config = _load_config(
        ContrastiveModelArguments(
            config_name=str(tmp_path),
            use_distributed_negatives=False,
        )
    )

    assert config.use_distributed_negatives is False


def test_fixed_train_batch_selection_is_deterministic():
    dataset = [{"value": index} for index in range(20)]

    def collator(samples):
        return {"values": [sample["value"] for sample in samples]}

    first = _build_fixed_train_batch(dataset, collator, sample_count=5, seed=7)
    second = _build_fixed_train_batch(dataset, collator, sample_count=5, seed=7)

    assert first == second
    assert len(first["values"]) == 5
