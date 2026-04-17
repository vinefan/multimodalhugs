from typing import Optional

from omegaconf import OmegaConf

from multimodalhugs.data.datasets.pose2text import Pose2TextDataConfig, Pose2TextDataset
from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.models.sign_clip.modeling_sign_clip import SignCLIPModel
from multimodalhugs.processors.sign_clip_processor import SignCLIPProcessor
from multimodalhugs.training_setup.setup_utils import (
    extract_tokenizer_info_from_processor_config,
    load_config,
    load_tokenizers,
    prepare_dataset,
    print_artifact_summary,
    resolve_setup_paths,
    resolve_update_choice,
    save_actor_paths,
    save_processor,
    update_configs,
)


def _build_sign_clip_config(model_cfg: dict, tokenizer=None) -> SignCLIPConfig:
    model_cfg = dict(model_cfg or {})
    model_cfg.pop("type", None)
    model_cfg.pop("model_name_or_path", None)

    if tokenizer is not None:
        for attr_name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            token_value = getattr(tokenizer, attr_name, None)
            if token_value is not None and model_cfg.get(attr_name) is None:
                model_cfg[attr_name] = token_value

    return SignCLIPConfig(**model_cfg)


def _build_and_save_sign_clip_model(config_path: str, output_dir: str, tokenizer=None) -> str:
    cfg = load_config(config_path)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    config = _build_sign_clip_config(model_cfg, tokenizer=tokenizer)
    model = SignCLIPModel(config)
    model_path = f"{output_dir}/model"
    model.save_pretrained(model_path)
    return model_path


def main(
    config_path: str,
    do_dataset: bool,
    do_processor: bool,
    do_model: bool,
    output_dir: Optional[str] = None,
    update_config: Optional[bool] = None,
    rebuild_dataset_from_scratch: bool = False,
):
    cfg = load_config(config_path)
    final_output_dir = resolve_setup_paths(cfg, output_dir)

    tok = pre_tok = None
    new = []

    data_path = None
    if do_dataset:
        print("\nSetting Up Dataset:\n")
        data_cfg = Pose2TextDataConfig(cfg)
        data_path = prepare_dataset(
            Pose2TextDataset,
            data_cfg,
            final_output_dir,
            rebuild_from_scratch=rebuild_dataset_from_scratch,
        )

    proc_path = None
    if do_processor:
        print("\nSetting Up Processor:\n")
        processor_cfg = getattr(cfg, "processor", None)
        tok_path, new_vocab = extract_tokenizer_info_from_processor_config(processor_cfg)
        if tok_path is None:
            raise ValueError(
                "SignCLIP setup requires a tokenizer_path/text_tokenizer_path in the processor config."
            )

        tok, pre_tok, new = load_tokenizers(tok_path, new_vocab)
        processor_kwargs = OmegaConf.to_container(processor_cfg, resolve=True) if processor_cfg else {}
        processor_kwargs.pop("processor_name_or_path", None)
        processor_kwargs.pop("tokenizer_path", None)
        processor_kwargs.pop("text_tokenizer_path", None)
        processor_kwargs.pop("new_vocabulary", None)
        processor_kwargs.pop("pipeline", None)
        processor_kwargs.pop("slots", None)

        proc = SignCLIPProcessor(
            tokenizer=tok,
            **processor_kwargs,
        )
        proc_path = save_processor(proc, final_output_dir)

    model_path = None
    if do_model:
        print("\nSetting Up Model:\n")
        if tok is None:
            processor_cfg = getattr(cfg, "processor", None)
            tok_path, new_vocab = extract_tokenizer_info_from_processor_config(processor_cfg)
            if tok_path is None:
                raise ValueError(
                    "Cannot determine tokenizer_path for SignCLIP model setup. "
                    "Set tokenizer_path/text_tokenizer_path in the processor config, or run with do_processor=True."
                )
            tok, pre_tok, new = load_tokenizers(tok_path, new_vocab)

        model_path = _build_and_save_sign_clip_model(config_path, final_output_dir, tokenizer=tok)

    should_update = resolve_update_choice(cfg, update_config)
    if should_update:
        update_configs(
            config_path,
            processor_path=proc_path,
            data_path=data_path,
            model_path=model_path,
        )
    else:
        print_artifact_summary(proc_path, model_path, data_path)

    save_actor_paths(final_output_dir, proc_path, data_path, model_path)
