from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContrastiveModelArguments:
    """
    Minimal argument scaffold for contrastive model configuration.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional pretrained model path for contrastive training."},
    )
    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional YAML config path for the contrastive task."},
    )
    processor_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional processor path used by the contrastive task."},
    )


@dataclass
class ContrastiveDataArguments:
    """
    Minimal argument scaffold for contrastive dataset inputs.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset identifier or local dataset path."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional training split file path."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional validation split file path."},
    )


@dataclass
class ContrastiveTrainingArguments:
    """
    Minimal argument scaffold for contrastive training control.
    """

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
