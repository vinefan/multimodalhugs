import torch
from transformers import BertTokenizerFast

from multimodalhugs.processors.sign_clip_processor import SignCLIPProcessor


def test_sign_clip_processor_outputs_expected_fields(tmp_path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "\n".join(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<en>", "<ase>", "hello", "world"]
        )
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path))

    processor = SignCLIPProcessor(tokenizer=tokenizer, reduce_holistic_poses=True)

    batch = [
        {
            "signal": torch.randn(4, 8),
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "<en> <ase>",
            "output": "hello world",
        },
        {
            "signal": torch.randn(2, 8),
            "signal_start": 0,
            "signal_end": 0,
            "encoder_prompt": "<en> <ase>",
            "output": "hello",
        },
    ]

    outputs = processor(batch=batch, batch_dict={}, return_tensors="pt")

    assert set(outputs.keys()) == {
        "sign_inputs",
        "sign_attention_mask",
        "input_ids",
        "attention_mask",
    }
    assert outputs["sign_inputs"].shape == (2, 4, 8)
    assert outputs["sign_attention_mask"].shape == (2, 4)
    assert outputs["input_ids"].shape[0] == 2
    assert outputs["attention_mask"].shape == outputs["input_ids"].shape
    decoded = tokenizer.batch_decode(outputs["input_ids"], skip_special_tokens=True)
    assert decoded[0] == "<en> <ase> hello world"
    assert decoded[1] == "<en> <ase> hello"


def test_sign_clip_processor_truncates_sign_inputs(tmp_path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"]))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path))
    processor = SignCLIPProcessor(tokenizer=tokenizer, max_sign_frames=4)

    outputs = processor(
        batch=[{"signal": torch.randn(6, 8), "output": "hello"}],
        batch_dict={},
        return_tensors="pt",
    )

    assert outputs["sign_inputs"].shape == (1, 4, 8)
    assert outputs["sign_attention_mask"].shape == (1, 4)
