import torch
from transformers import BertTokenizerFast

from multimodalhugs.data.datacollators.contrastive_datacollator import (
    DataCollatorContrastive,
)
from multimodalhugs.processors.sign_clip_processor import SignCLIPProcessor


def test_contrastive_datacollator_returns_model_inputs(tmp_path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "\n".join(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<en>", "<ase>", "hello", "world"]
        )
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path))
    processor = SignCLIPProcessor(tokenizer=tokenizer, reduce_holistic_poses=True)
    collator = DataCollatorContrastive(processor=processor)

    samples = [
        {
            "signal": torch.randn(4, 8),
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "<en> <ase>",
            "decoder_prompt": "",
            "output": "hello world",
        },
        {
            "signal": torch.randn(2, 8),
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "<en> <ase>",
            "decoder_prompt": "",
            "output": "hello",
        },
    ]

    batch = collator(samples)

    assert set(batch.keys()) == {
        "sign_inputs",
        "sign_attention_mask",
        "input_ids",
        "attention_mask",
    }
    assert batch["sign_inputs"].shape == (2, 4, 8)
    assert batch["sign_attention_mask"].shape == (2, 4)
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_contrastive_datacollator_can_include_metadata(tmp_path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "\n".join(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<en>", "<ase>", "hello", "world"]
        )
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path))
    processor = SignCLIPProcessor(tokenizer=tokenizer, reduce_holistic_poses=True)
    collator = DataCollatorContrastive(processor=processor, include_metadata=True, metadata_keys=["idx"])

    samples = [
        {
            "idx": 1,
            "signal": torch.randn(4, 8),
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "<en> <ase>",
            "decoder_prompt": "",
            "output": "hello world",
        },
        {
            "idx": 2,
            "signal": torch.randn(2, 8),
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "<en> <ase>",
            "decoder_prompt": "",
            "output": "hello",
        },
    ]

    batch = collator(samples)

    assert batch["idx"] == [1, 2]
