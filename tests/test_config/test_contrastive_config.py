from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.tasks.contrastive.config_classes import ContrastiveModelArguments
from multimodalhugs.tasks.contrastive.contrastive_training import (
    ContrastiveTrainer,
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


def test_fixed_train_loss_uses_fractional_trainer_epoch():
    trainer = ContrastiveTrainer.__new__(ContrastiveTrainer)
    trainer.fixed_train_batch = {"input_ids": [1]}
    trainer.fixed_train_loss_steps = 100
    trainer.fixed_train_loss_at_epoch_end = True
    trainer._last_fixed_train_loss_step = None
    trainer.state = type("State", (), {"global_step": 101, "epoch": 0.025})()

    assert not trainer._should_measure_fixed_train_loss(trainer.state.epoch)

    trainer.state.global_step = 100
    assert trainer._should_measure_fixed_train_loss(trainer.state.epoch)

    trainer.state.global_step = 4024
    trainer.state.epoch = 1.0
    assert trainer._should_measure_fixed_train_loss(trainer.state.epoch)
