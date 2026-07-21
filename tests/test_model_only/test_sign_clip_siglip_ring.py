import os
import sys
import tempfile

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.models.sign_clip.modeling_sign_clip import SignCLIPModel


def _build_ring_test_model():
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
        projection_dim=3,
        contrastive_loss_type="siglip",
        logit_scale_init_value=0.0,
        logit_bias_init_value=-1.0,
        siglip_distributed_implementation="ring",
    )
    return SignCLIPModel(config)


def _ring_worker(rank, world_size, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        all_sign = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0], [0.0, 0.5, 1.0]]
        )
        all_text = torch.tensor(
            [[0.8, 0.1, 0.4], [0.1, 0.9, 0.3], [0.4, 0.2, 0.9], [0.2, 0.6, 0.8]]
        )
        local_batch_size = all_sign.size(0) // world_size
        start = rank * local_batch_size
        stop = start + local_batch_size

        local_sign = all_sign[start:stop].clone().requires_grad_(True)
        local_text = all_text[start:stop].clone().requires_grad_(True)
        model = _build_ring_test_model()

        ring_loss, local_logits, _ = model._compute_siglip_ring_loss(local_sign, local_text)
        ring_loss.backward()
        ring_sign_grad = local_sign.grad.clone()
        ring_text_grad = local_text.grad.clone()

        reference_sign = all_sign.clone().requires_grad_(True)
        reference_text = all_text.clone().requires_grad_(True)
        reference_logits = reference_sign @ reference_text.T - 1.0
        pair_labels = 2.0 * torch.eye(reference_logits.size(0)) - 1.0
        reference_loss = -torch.nn.functional.logsigmoid(pair_labels * reference_logits).sum()
        reference_loss = reference_loss / local_batch_size
        reference_loss.backward()

        local_reference_logits = reference_logits[start:stop, start:stop]
        torch.testing.assert_close(local_logits, local_reference_logits)
        torch.testing.assert_close(ring_sign_grad, reference_sign.grad[start:stop])
        torch.testing.assert_close(ring_text_grad, reference_text.grad[start:stop])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="PyTorch multiprocessing conflicts with macOS MPS initialization in the test process.",
)
def test_siglip_ring_matches_full_matrix_feature_gradients():
    if not dist.is_available():
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        init_method = f"file://{os.path.join(temp_dir, 'distributed-init')}"
        mp.spawn(_ring_worker, args=(2, init_method), nprocs=2, join=True)
