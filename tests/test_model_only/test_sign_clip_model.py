import torch

from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.models.sign_clip.modeling_sign_clip import SignCLIPModel


def _build_tiny_sign_clip_config():
    encoder_config = {
        "vocab_size": 100,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 64,
        "type_vocab_size": 2,
        "pad_token_id": 0,
        "cls_token_id": 1,
        "sep_token_id": 2,
    }

    return SignCLIPConfig(
        sign_encoder_type="bert",
        sign_encoder_config=encoder_config,
        text_encoder_type="bert",
        text_encoder_config=encoder_config,
        sign_input_dim=16,
        projection_dim=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        use_projection=True,
        projection_l2_norm=True,
        return_dict=True,
    )


def test_sign_clip_model_initialization():
    config = _build_tiny_sign_clip_config()
    model = SignCLIPModel(config)

    assert model.sign_encoder is not None
    assert model.text_encoder is not None
    assert model.sign_token_mlp is not None
    assert model.logit_scale is not None
    assert model.sign_projection is not None
    assert model.text_projection is not None


def test_sign_clip_model_forward():
    torch.manual_seed(0)

    config = _build_tiny_sign_clip_config()
    model = SignCLIPModel(config)
    model.eval()

    sign_inputs = torch.randn(2, 5, 16)
    sign_attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    input_ids = torch.tensor(
        [
            [1, 11, 12, 13, 2, 0],
            [1, 21, 22, 2, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    outputs = model(
        sign_inputs=sign_inputs,
        sign_attention_mask=sign_attention_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_loss=True,
    )

    assert outputs.loss is not None
    assert outputs.logits_per_sign.shape == (2, 2)
    assert outputs.logits_per_text.shape == (2, 2)
    assert outputs.sign_embeds.shape == (2, 32)
    assert outputs.text_embeds.shape == (2, 32)
