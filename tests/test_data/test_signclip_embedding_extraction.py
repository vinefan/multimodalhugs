from types import SimpleNamespace

import pytest
import torch

from multimodalhugs.tasks.contrastive.signclip_embedding_extraction import (
    _truncate_sign_batch_to_model_limit,
    _validate_sign_feature_dimension,
    sign_collator,
)


def _model_with_position_limit(max_positions: int):
    return SimpleNamespace(
        config=SimpleNamespace(sign_input_dim=4),
        sign_encoder=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=max_positions),
        )
    )


def test_truncates_pose_frames_while_reserving_cls_and_sep_positions():
    batch = {
        "sign_inputs": torch.randn(2, 7, 4),
        "sign_attention_mask": torch.ones(2, 7, dtype=torch.long),
    }

    truncated, was_truncated = _truncate_sign_batch_to_model_limit(
        _model_with_position_limit(6),
        batch,
    )

    assert was_truncated is True
    assert truncated["sign_inputs"].shape == (2, 4, 4)
    assert truncated["sign_attention_mask"].shape == (2, 4)


def test_keeps_pose_frames_within_model_limit():
    batch = {
        "sign_inputs": torch.randn(2, 4, 4),
        "sign_attention_mask": torch.ones(2, 4, dtype=torch.long),
    }

    unchanged, was_truncated = _truncate_sign_batch_to_model_limit(
        _model_with_position_limit(6),
        batch,
    )

    assert was_truncated is False
    assert torch.equal(unchanged["sign_inputs"], batch["sign_inputs"])
    assert torch.equal(unchanged["sign_attention_mask"], batch["sign_attention_mask"])


def test_accepts_matching_pose_feature_dimension():
    batch = {
        "sign_inputs": torch.randn(2, 7, 4),
        "sign_attention_mask": torch.ones(2, 7, dtype=torch.long),
    }

    _validate_sign_feature_dimension(_model_with_position_limit(6), batch)


def test_rejects_pose_feature_dimension_different_from_training():
    batch = {
        "sign_inputs": torch.randn(2, 7, 8),
        "sign_attention_mask": torch.ones(2, 7, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="expected 4 features per frame, received 8"):
        _validate_sign_feature_dimension(_model_with_position_limit(6), batch)


class _ProcessorWithUnreadableSample:
    def _signal_to_tensor(self, signal, signal_start, signal_end):
        if signal == "bad.pose":
            raise OSError("truncated pose")
        return torch.ones(3, 4)


def test_sign_collator_reports_and_skips_unreadable_samples():
    collate = sign_collator(_ProcessorWithUnreadableSample(), "skip")

    result = collate(
        [
            (4, {"signal": "good.pose"}),
            (9, {"signal": "bad.pose"}),
        ]
    )

    assert result["dataset_indices"] == [4]
    assert result["model_inputs"]["sign_inputs"].shape == (1, 3, 4)
    assert result["bad_samples"] == [
        {
            "dataset_index": 9,
            "signal": "bad.pose",
            "error_type": "OSError",
            "error": "truncated pose",
        }
    ]


def test_sign_collator_error_includes_unreadable_sample_path():
    collate = sign_collator(_ProcessorWithUnreadableSample(), "error")

    with pytest.raises(RuntimeError, match="bad\\.pose.*truncated pose"):
        collate([(9, {"signal": "bad.pose"})])
