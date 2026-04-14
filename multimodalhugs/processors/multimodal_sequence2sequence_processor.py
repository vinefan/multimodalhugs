import inspect
import logging

from typing import List, Dict, Any, Optional, Callable

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

logger = logging.getLogger(__name__)


class MultimodalSequence2SequenceProcessor(ProcessorMixin):
    name = "multimodal_sequence2sequence_processor"
    attributes = ["frame_preprocessor", "tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    frame_preprocessor_class = "BaseImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs: List[str] = ["obtainables_list"]

    def __init__(
        self,
        frame_preprocessor: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        processor_name_or_path: Optional[Any] = None,
        text_tokenizer_path: Optional[Any] = None,
        new_vocabulary: Optional[Any] = None,
        **kwargs,
    ):
        obtainables_list = kwargs.pop("obtainables_list", None)
        self.obtainables_list = obtainables_list

        valid_super_args = set(inspect.signature(super().__init__).parameters)
        used_kwargs = {k: v for k, v in kwargs.items() if k in valid_super_args}
        unused_kwargs = {k: v for k, v in kwargs.items() if k not in valid_super_args}

        if unused_kwargs:
            logger.warning(
                " The following kwargs are not used by the processor and will be ignored: %s",
                list(unused_kwargs.keys()),
            )

        if self.__class__._transform_get_items_output is MultimodalSequence2SequenceProcessor._transform_get_items_output:
            logger.warning(
                " %s does not override `_transform_get_items_output()`. This method should define a "
                "dataset-level transformation applied during iteration via `dataset.with_transform()`. "
                "Not overriding it may result in inefficiencies.",
                self.__class__.__name__,
            )

        if frame_preprocessor is None:
            super().__init__(
                tokenizer=tokenizer,
                **used_kwargs,
            )
        else:
            super().__init__(
                frame_preprocessor=frame_preprocessor,
                tokenizer=tokenizer,
                **used_kwargs,
            )

    def process_prompts(self, prompts):
        tokenized_output = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        padded_prompts = tokenized_output["input_ids"]
        prompt_length_padding_mask = tokenized_output["attention_mask"]

        return padded_prompts, prompt_length_padding_mask

    def get_obtainables(self):
        if self.obtainables_list is not None:
            obtainables = [getattr(self, method_name) for method_name in self.obtainables_list]
        else:
            obtainables = [getattr(self, method_name) for method_name in dir(self) if method_name.startswith("_obtain_")]

        obtain_whatever_method = getattr(self, "_obtain_whatever", None)
        if obtain_whatever_method and len(obtainables) > 1:
            obtainables = [method for method in obtainables if method != obtain_whatever_method]

        return obtainables

    def get_langtok(self, langtok):
        langtok_idx = None
        if self.tokenizer is not None:
            langtok_idx = self.tokenizer.convert_tokens_to_ids(langtok)
        return langtok_idx

    def _obtain_whatever(self, batch, **kwargs):
        raise NotImplementedError("_obtain_<whatever> methods must be implemented by the child class.")

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        raise NotImplementedError("_obtain_multimodal_input_and_masks method must be implemented by the child class.")

    def _obtain_encoder_prompt(self, batch, **kwargs):
        padded_prompts, encoder_prompt_length_padding_mask = self.process_prompts([sample["encoder_prompt"] for sample in batch])

        return {
            "encoder_prompt": padded_prompts,
            "encoder_prompt_length_padding_mask": encoder_prompt_length_padding_mask,
        }, kwargs

    def _obtain_decoder_prompt(self, batch, **kwargs):
        padded_prompts, decoder_prompt_length_padding_mask = self.process_prompts([sample["decoder_prompt"] for sample in batch])

        return {
            "decoder_input_ids": padded_prompts,
            "decoder_attention_mask": decoder_prompt_length_padding_mask,
        }, kwargs

    def __call__(
        self,
        batch: List[Dict[str, Any]],
        batch_dict: Optional[Dict[str, Any]] = {},
        **kwargs,
    ) -> BatchFeature:
        for obtain_method in self.get_obtainables():
            obtained_dict, kwargs = obtain_method(batch, **kwargs)
            for k, v in obtained_dict.items():
                if k == "decoder_input_ids" and k in batch_dict:
                    continue
                batch_dict[k] = v

        return BatchFeature(batch_dict)

    def _transform_get_items_output(self, batch):
        return batch
