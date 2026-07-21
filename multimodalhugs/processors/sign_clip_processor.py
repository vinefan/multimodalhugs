import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.processors.multimodal_sequence2sequence_processor import MultimodalSequence2SequenceProcessor
from multimodalhugs.processors.utils import frame_skipping
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, pose_hide_legs

logger = logging.getLogger(__name__)


class SignCLIPProcessor(MultimodalSequence2SequenceProcessor):
    name = "sign_clip_processor"
    attributes = ["tokenizer"]
    model_input_names = ["sign_inputs", "sign_attention_mask", "input_ids", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        reduce_holistic_poses: bool = True,
        skip_frames_stride: Optional[int] = None,
        pose_components: Optional[list[str]] = None,
        max_sign_frames: Optional[int] = None,
        **kwargs,
    ):
        obtainables_list = kwargs.pop(
            "obtainables_list",
            ["_obtain_multimodal_input_and_masks", "_obtain_text_inputs_and_masks"],
        )
        super().__init__(
            tokenizer=tokenizer,
            obtainables_list=obtainables_list,
            **kwargs,
        )
        self.reduce_holistic_poses = reduce_holistic_poses
        self.skip_frames_stride = skip_frames_stride
        self.pose_components = pose_components
        self.max_sign_frames = max_sign_frames

    def _signal_to_tensor(
        self,
        signal: Union[str, Path, np.ndarray, torch.Tensor, list],
        signal_start: int = 0,
        signal_end: int = 0,
    ) -> torch.Tensor:
        if isinstance(signal, torch.Tensor):
            tensor = signal
        elif isinstance(signal, np.ndarray):
            tensor = torch.from_numpy(signal)
        elif isinstance(signal, (str, Path)):
            signal_path = Path(signal)
            if signal_path.suffix == ".pose":
                with open(signal_path, "rb") as pose_file:
                    pose = Pose.read(
                        pose_file,
                        start_time=signal_start or None,
                        end_time=signal_end or None,
                    )

                if self.pose_components:
                    pose = pose.normalize()
                    if "POSE_LANDMARKS" in self.pose_components:
                        pose_hide_legs(pose)
                    pose = pose.get_components(self.pose_components)
                else:
                    pose_hide_legs(pose)
                    if self.reduce_holistic_poses:
                        pose = reduce_holistic(pose)
                    pose = pose.normalize()
                tensor = pose.torch().body.data.zero_filled()
                tensor = tensor.contiguous().view(tensor.size(0), -1)
            else:
                tensor = torch.from_numpy(np.load(signal_path))
        elif isinstance(signal, list):
            tensor = torch.tensor(signal, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported type for signal: {type(signal)}")

        tensor = tensor.float()
        if self.skip_frames_stride is not None:
            tensor = frame_skipping(x=tensor, t_dim=0, stride=self.skip_frames_stride)
        if self.max_sign_frames is not None:
            tensor = tensor[: self.max_sign_frames]
        return tensor

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        tensor_sequences = [
            self._signal_to_tensor(
                sample["signal"],
                sample.get("signal_start") or 0,
                sample.get("signal_end") or 0,
            )
            for sample in batch
        ]
        sign_inputs, sign_attention_mask = pad_and_create_mask(tensor_sequences)
        return {
            "sign_inputs": sign_inputs,
            "sign_attention_mask": sign_attention_mask,
        }, kwargs

    def _obtain_text_inputs_and_masks(self, batch, **kwargs):
        text_inputs = []
        for sample in batch:
            encoder_prompt = (sample.get("encoder_prompt") or "").strip()
            output = (sample.get("output") or "").strip()
            text_inputs.append(f"{encoder_prompt} {output}".strip())

        tokenized_output = self.tokenizer(
            text_inputs,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized_output["input_ids"],
            "attention_mask": tokenized_output["attention_mask"],
        }, kwargs

    def _transform_get_items_output(self, batch):
        batch["signal"] = [
            self._signal_to_tensor(signal, start or 0, end or 0)
            for signal, start, end in zip(
                batch["signal"],
                batch.get("signal_start", [0] * len(batch["signal"])),
                batch.get("signal_end", [0] * len(batch["signal"])),
            )
        ]
        return batch
