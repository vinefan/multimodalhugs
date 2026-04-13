from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class DataCollatorContrastive:
    """
    Lightweight data collator for contrastive multimodal tasks.

    The collator delegates semantic input preparation to the processor and only
    returns a batch that can be consumed directly by `SignCLIPModel`.
    """

    processor: Any
    return_tensors: str = "pt"
    include_metadata: bool = False
    metadata_keys: Optional[List[str]] = None

    def __init__(
        self,
        processor: Any,
        return_tensors: str = "pt",
        include_metadata: bool = False,
        metadata_keys: Optional[List[str]] = None,
    ):
        self.processor = processor
        self.return_tensors = return_tensors
        self.include_metadata = include_metadata
        self.metadata_keys = metadata_keys or ["idx", "signal", "output", "encoder_prompt"]

    def __call__(
        self,
        samples: List[Dict[str, Union[List[int], torch.Tensor, str]]]
    ) -> Dict[str, Any]:
        batch = self.processor(
            batch=samples,
            batch_dict={},
            return_tensors=self.return_tensors,
        )

        if self.include_metadata:
            for key in self.metadata_keys:
                values = [sample[key] for sample in samples if key in sample]
                if len(values) == len(samples):
                    batch[key] = values

        return batch
