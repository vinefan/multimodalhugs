from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ContrastiveModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained SignCLIP checkpoint."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional config path if different from model_name_or_path."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to cache pretrained model files."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model revision to use when loading remote artifacts."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "Authentication token used to access remote files."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading Hugging Face artifacts."},
    )


@dataclass
class ContrastiveProcessorArguments:
    processor_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained SignCLIP processor."},
    )


@dataclass
class ContrastiveDataArguments:
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a dataset saved with datasets.save_to_disk()."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on the number of training samples."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on the number of evaluation samples."},
    )
    train_ordering_strategy: str = field(
        default="default",
        metadata={"help": "Training-set ordering strategy. Supported values: `default`, `fairseq_round_robin`."},
    )


@dataclass
class ExtraArguments:
    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML config file."},
    )
    setup_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional setup dir used to infer missing actor paths."},
    )


@dataclass
class ContrastiveTrainingArguments(TrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "Number of eval calls with no improvement before early stopping."},
    )
    output_dir: str = field(
        default="outputs/contrastive",
        metadata={"help": "Directory used to store contrastive task outputs."},
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "Whether to run training."},
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation."},
    )
    run_retrieval_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run dataset-wide retrieval evaluation in addition to eval_loss."},
    )
    experiment_index_path: Optional[str] = field(
        default="experiments/signclip_runs.jsonl",
        metadata={"help": "Local JSONL file used to append one summary record per experiment run."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Weights & Biases project name. When set, contrastive runs are reported to W&B."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Weights & Biases entity/team name."},
    )
    wandb_tags: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated Weights & Biases tags for this run."},
    )
