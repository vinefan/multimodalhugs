from types import SimpleNamespace

import pytest
import torch

from multimodalhugs.tasks.contrastive.signclip_embedding_extraction import (
    _truncate_sign_batch_to_model_limit,
    _validate_sign_feature_dimension,
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
