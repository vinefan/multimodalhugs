#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import datasets
import transformers
from datasets import load_from_disk
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoConfig,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

from multimodalhugs.data.datacollators.contrastive_datacollator import DataCollatorContrastive
from multimodalhugs.models.sign_clip.configuration_sign_clip import SignCLIPConfig
from multimodalhugs.models.sign_clip.modeling_sign_clip import SignCLIPModel
from multimodalhugs.processors.sign_clip_processor import SignCLIPProcessor
from multimodalhugs.tasks.contrastive.config_classes import (
    ContrastiveDataArguments,
    ContrastiveModelArguments,
    ContrastiveProcessorArguments,
    ContrastiveTrainingArguments,
    ExtraArguments,
)
from multimodalhugs.tasks.contrastive.retrieval_eval import (
    collect_retrieval_outputs,
    compute_retrieval_metrics,
)
from multimodalhugs.tasks.translation.utils import (
    ensure_train_output_dir,
    merge_config_and_command_args,
    resolve_missing_arg,
)

logger = logging.getLogger(__name__)


class ContrastiveTrainer(Trainer):
    def __init__(self, *args, train_ordering_strategy: str = "default", **kwargs):
        super().__init__(*args, **kwargs)
        self.train_ordering_strategy = train_ordering_strategy

    def _get_train_sampler(self):
        if self.train_ordering_strategy == "fairseq_round_robin":
            if self.train_dataset is None or not hasattr(self.train_dataset, "__len__"):
                return None
            logger.info("Using SequentialSampler to preserve fairseq_round_robin training order.")
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()


def _apply_train_ordering(split_dataset, strategy: str):
    if strategy == "default":
        return split_dataset

    if strategy != "fairseq_round_robin":
        raise ValueError(
            f"Unsupported train ordering strategy: {strategy}. "
            "Supported values are `default` and `fairseq_round_robin`."
        )

    if "output" not in split_dataset.column_names:
        raise ValueError("fairseq_round_robin ordering requires an `output` column in the training dataset.")

    grouped_indices = defaultdict(list)
    for index, output in enumerate(split_dataset["output"]):
        grouped_indices[output].append(index)

    if not grouped_indices:
        return split_dataset

    max_group_size = max(len(indices) for indices in grouped_indices.values())
    expanded_indices = []
    for round_index in range(max_group_size):
        for group in grouped_indices.values():
            expanded_indices.append(group[round_index % len(group)])

    logger.info(
        "Applied fairseq_round_robin train ordering across %s groups, expanding %s samples to %s positions.",
        len(grouped_indices),
        len(split_dataset),
        len(expanded_indices),
    )
    return split_dataset.select(expanded_indices)


def _parse_args():
    parser = HfArgumentParser(
        (
            ExtraArguments,
            ContrastiveModelArguments,
            ContrastiveProcessorArguments,
            ContrastiveDataArguments,
            ContrastiveTrainingArguments,
        )
    )
    extra_args, model_args, processor_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if extra_args.config_path:
        model_args = merge_config_and_command_args(
            extra_args.config_path,
            ContrastiveModelArguments,
            "model",
            model_args,
            sys.argv[1:],
        )
        processor_args = merge_config_and_command_args(
            extra_args.config_path,
            ContrastiveProcessorArguments,
            "processor",
            processor_args,
            sys.argv[1:],
        )
        data_args = merge_config_and_command_args(
            extra_args.config_path,
            ContrastiveDataArguments,
            "data",
            data_args,
            sys.argv[1:],
        )
        training_args = merge_config_and_command_args(
            extra_args.config_path,
            ContrastiveTrainingArguments,
            "training",
            training_args,
            sys.argv[1:],
        )

    return extra_args, model_args, processor_args, data_args, training_args


def _setup_logging(training_args: ContrastiveTrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        training_args.parallel_mode.value == "distributed",
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)


def _configure_wandb(training_args: ContrastiveTrainingArguments):
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
        if training_args.wandb_entity:
            os.environ["WANDB_ENTITY"] = training_args.wandb_entity
        if training_args.wandb_tags:
            os.environ["WANDB_TAGS"] = training_args.wandb_tags
        if not training_args.report_to or training_args.report_to == ["none"]:
            training_args.report_to = ["wandb"]
    elif isinstance(training_args.report_to, list) and "wandb" in training_args.report_to:
        logger.info("W&B reporting requested via report_to.")


def _safe_git_value(args):
    try:
        return (
            subprocess.check_output(args, cwd=os.getcwd(), stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
    except Exception:  # pragma: no cover - best effort metadata only
        return None


def _to_builtin(value):
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _append_experiment_record(
    *,
    extra_args: ExtraArguments,
    model_args: ContrastiveModelArguments,
    processor_args: ContrastiveProcessorArguments,
    data_args: ContrastiveDataArguments,
    training_args: ContrastiveTrainingArguments,
    train_metrics: dict,
    eval_metrics: dict,
):
    if not training_args.experiment_index_path:
        return

    index_path = Path(training_args.experiment_index_path)
    if not index_path.is_absolute():
        index_path = Path(os.getcwd()) / index_path
    index_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": training_args.run_name,
        "output_dir": training_args.output_dir,
        "git_branch": _safe_git_value(["git", "branch", "--show-current"]),
        "git_commit": _safe_git_value(["git", "rev-parse", "HEAD"]),
        "config_path": extra_args.config_path,
        "setup_path": extra_args.setup_path,
        "model_name_or_path": model_args.model_name_or_path,
        "use_distributed_negatives": model_args.use_distributed_negatives,
        "processor_name_or_path": processor_args.processor_name_or_path,
        "dataset_dir": data_args.dataset_dir,
        "train_ordering_strategy": data_args.train_ordering_strategy,
        "run_retrieval_eval": training_args.run_retrieval_eval,
        "report_to": _to_builtin(training_args.report_to),
        "wandb_project": training_args.wandb_project,
        "wandb_entity": training_args.wandb_entity,
        "wandb_tags": training_args.wandb_tags,
        "train_metrics": _to_builtin(train_metrics),
        "eval_metrics": _to_builtin(eval_metrics),
    }

    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Appended experiment summary to %s", index_path)


def _detect_last_checkpoint(training_args: ContrastiveTrainingArguments) -> Optional[str]:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                "Checkpoint detected, resuming training at %s. To avoid this behavior, change the "
                "`--output_dir` or add `--overwrite_output_dir` to train from scratch.",
                last_checkpoint,
            )
    return last_checkpoint


def _load_config(model_args: ContrastiveModelArguments) -> SignCLIPConfig:
    config_source = model_args.config_name or model_args.model_name_or_path
    if config_source is None:
        return SignCLIPConfig()

    config = AutoConfig.from_pretrained(
        config_source,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if not isinstance(config, SignCLIPConfig):
        config = SignCLIPConfig(**config.to_dict())
    if model_args.use_distributed_negatives is not None:
        config.use_distributed_negatives = model_args.use_distributed_negatives
        logger.info(
            "Overriding use_distributed_negatives=%s",
            config.use_distributed_negatives,
        )
    return config


def _load_processor(processor_args: ContrastiveProcessorArguments) -> SignCLIPProcessor:
    if not processor_args.processor_name_or_path:
        raise ValueError("You must specify processor_name_or_path in the config or on the command line.")
    return SignCLIPProcessor.from_pretrained(processor_args.processor_name_or_path)


def _load_model(model_args: ContrastiveModelArguments, config: SignCLIPConfig) -> SignCLIPModel:
    if model_args.model_name_or_path:
        return SignCLIPModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    return SignCLIPModel(config)


def _prepare_dataset(split_dataset, max_samples: Optional[int], processor: SignCLIPProcessor):
    dataset = split_dataset.with_transform(processor._transform_get_items_output)
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    return dataset


def _run_retrieval_eval(
    model: SignCLIPModel,
    eval_dataset,
    processor: SignCLIPProcessor,
    training_args: ContrastiveTrainingArguments,
):
    batch_size = training_args.per_device_eval_batch_size
    retrieval_outputs = collect_retrieval_outputs(
        model=model,
        dataset=eval_dataset,
        processor=processor,
        batch_size=batch_size,
        num_workers=training_args.dataloader_num_workers,
    )
    return compute_retrieval_metrics(
        sign_embeds=retrieval_outputs["sign_embeds"],
        text_embeds=retrieval_outputs["text_embeds"],
        texts=retrieval_outputs["texts"],
        direction=training_args.retrieval_eval_direction,
    )


def main():
    extra_args, model_args, processor_args, data_args, training_args = _parse_args()

    resolve_missing_arg(
        model_args,
        "model_name_or_path",
        training_args.output_dir,
        extra_args.setup_path if hasattr(extra_args, "setup_path") else None,
    )
    resolve_missing_arg(
        processor_args,
        "processor_name_or_path",
        training_args.output_dir,
        extra_args.setup_path if hasattr(extra_args, "setup_path") else None,
    )
    resolve_missing_arg(
        data_args,
        "dataset_dir",
        training_args.output_dir,
        extra_args.setup_path if hasattr(extra_args, "setup_path") else None,
    )

    training_args.output_dir = ensure_train_output_dir(training_args.output_dir)
    setattr(training_args, "remove_unused_columns", False)
    _configure_wandb(training_args)

    send_example_telemetry("run_contrastive", model_args, data_args)
    _setup_logging(training_args)

    last_checkpoint = _detect_last_checkpoint(training_args)
    set_seed(training_args.seed)

    if data_args.dataset_dir is None:
        raise ValueError("You must specify dataset_dir in the config or on the command line.")
    raw_datasets = load_from_disk(data_args.dataset_dir)

    processor = _load_processor(processor_args)
    config = _load_config(model_args)
    model = _load_model(model_args, config)

    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        ordered_train_dataset = _apply_train_ordering(raw_datasets["train"], data_args.train_ordering_strategy)
        train_dataset = _prepare_dataset(ordered_train_dataset, data_args.max_train_samples, processor)

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = _prepare_dataset(raw_datasets["validation"], data_args.max_eval_samples, processor)

    if not training_args.do_train and not training_args.do_eval:
        logger.info("There is nothing to do. Please pass `do_train` and/or `do_eval`.")
        return {}

    data_collator = DataCollatorContrastive(processor=processor)

    callbacks = []
    if training_args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        train_ordering_strategy=data_args.train_ordering_strategy,
    )

    metrics_result = {}
    train_metrics_result = {}

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        logger.info("Resuming training from: %s", checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        train_metrics_result = dict(metrics)
        if train_dataset is not None:
            metrics["train_samples"] = len(train_dataset)
            train_metrics_result["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics_result = trainer.evaluate()
        if eval_dataset is not None:
            metrics_result["eval_samples"] = len(eval_dataset)
        if training_args.run_retrieval_eval and eval_dataset is not None:
            retrieval_metrics = _run_retrieval_eval(
                model=model,
                eval_dataset=eval_dataset,
                processor=processor,
                training_args=training_args,
            )
            metrics_result.update(retrieval_metrics)
        trainer.log_metrics("eval", metrics_result)
        trainer.save_metrics("eval", metrics_result)

    _append_experiment_record(
        extra_args=extra_args,
        model_args=model_args,
        processor_args=processor_args,
        data_args=data_args,
        training_args=training_args,
        train_metrics=train_metrics_result,
        eval_metrics=metrics_result,
    )

    return metrics_result


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
