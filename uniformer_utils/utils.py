# This code has been adapted from the GitHub repository 'https://github.com/potamides/uniformers'
# Portions of the original code have been modified to fit the specific requirements
# of this project. Credit goes to the original authors for their contributions.

from abc import ABC, abstractmethod
from functools import cached_property
from os.path import isdir, isfile, join
from re import sub
from typing import Any, Dict, List, Optional, Tuple, Union

from numpy import where
from torch import Tensor, cat, full, nn

from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from transformers.training_args import TrainingArguments
from transformers.utils import logging

logger = logging.get_logger("transformers")

LM_LEARNING_RATES = {
    "base": 6e-4,
    "medium": 3e-4,
    "large": 2.5e-4,
    "xl": 2e-4,
}


class GlobalBatchTrainingArguments(TrainingArguments):
    """
    TrainingArguments class which evenly distributes batch_size on available
    GPUs under distributed training (DistributedDataParallel). Normal
    TrainingArguments use same batch_size on each GPU. (see
    https://discuss.pytorch.org/t/should-we-split-batch-size-according-to-ngpu-per-node-when-distributeddataparallel/72769/15)
    This should also work for DataParallel which does splitting on its own (see
    https://discuss.pytorch.org/t/a-question-concerning-batchsize-and-multiple-gpus-in-pytorch/33767).
    Additionally, batch_size is scaled according to gradient accumulation
    steps.
    """

    def __init__(
        self,
        global_train_batch_size=8,
        global_eval_batch_size=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_train_batch_size = global_train_batch_size
        self.global_eval_batch_size = global_eval_batch_size
        self.per_device_train_batch_size = self._scale_batch_size(
            global_train_batch_size
        )
        self.per_device_eval_batch_size = self._scale_batch_size(
            global_eval_batch_size)
        if self.world_size > 1:
            logger.info(
                f"Dividing batches equally on {self.world_size} processes.")

    def _scale_batch_size(self, batch_size) -> int:
        scaled_batch_size, remainder = divmod(
            batch_size,
            self.world_size * self.gradient_accumulation_steps,
        )
        if remainder != 0:
            raise ValueError(
                "`batch_size` must be divisible by number of processes times gradient accumulation steps."
            )
        return scaled_batch_size


class PoetryLMTrainingArguments(GlobalBatchTrainingArguments):
    def __init__(
        self,
        eval_multiplier=75,
        max_length=384,
        num_beams=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eval_multiplier = eval_multiplier
        self.max_length = max_length
        self.num_beams = num_beams


class AbstractPoetryLMTrainer(Trainer, ABC):
    @abstractmethod
    def __init__(
        self,
        model,
        tokenizer,
        eval_multiplier,
        output_dir,
        # https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803
        fp16=False,
        bf16=True,
        tf32=False,
        batch_size=128,
        overwrite_output_dir=False,
        # only change below stuff when model doesn't fit into memory (see
        # https://huggingface.co/docs/transformers/performance)
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        test_run=False,
        **trainer_args
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.trainer_args = trainer_args
        self.patch_tokenizer()

        if self.parameters < 280 * 10**6:
            learning_rate = "base"
        elif self.parameters < 770 * 10**6:
            learning_rate = "medium"
        elif self.parameters < 1550 * 10**6:
            learning_rate = "large"
        else:
            learning_rate = "xl"
        logger.info(
            f"Using learning rate for training {learning_rate} language models."
        )

        # interesting resource: https://huggingface.co/course/chapter7/6?fw=pt
        self.args = PoetryLMTrainingArguments(
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            learning_rate=LM_LEARNING_RATES[learning_rate],
            num_train_epochs=1 if test_run else 4,
            weight_decay=0.1,
            warmup_ratio=0.01,
            # eval_multiplier=eval_multiplier,
            global_train_batch_size=batch_size,
            global_eval_batch_size=batch_size,
            fp16=fp16,
            bf16=bf16,
            tf32=tf32,
            save_total_limit=1,
            overwrite_output_dir=overwrite_output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            ddp_find_unused_parameters=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20 if test_run else 250,
            logging_first_step=True,
            output_dir=output_dir,
            report_to="wandb",
        )

    @abstractmethod
    def compute_metrics(self, p):
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        # replace dynamic padding index with true padding index
        preds = self.decode(
            where(preds == -100, self.tokenizer.pad_token_id, preds), batch=True)
        # remove text after eos token (if token exists)
        preds = [pred[:pred.find(self.tokenizer.eos_token, len(
            self.tokenizer.eos_token))] for pred in preds]

        return preds

    @abstractmethod
    def patch_tokenizer(self):
        # melzohbi: removed this as it is not needed
        # if isinstance(self.tokenizer, ByGPT5Tokenizer):
        #     self.tokenizer.add_bos_token = False
        #     self.tokenizer.add_eos_token = False
        #     self.tokenizer.bos_token = self.tokenizer.eos_token
        if isinstance(self.tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
            self.tokenizer.add_bos_token = False  # pyright: ignore
        elif isinstance(self.tokenizer, (T5Tokenizer, T5TokenizerFast)):
            # sentencepiece replaces \n, so we need our own symbol (e.g., as sep_token)
            self.tokenizer.add_special_tokens(
                {"sep_token": "\n"})  # pyright: ignore
            self.model.resize_token_embeddings(len(self.tokenizer))

    @abstractmethod
    def load_dataset(self):
        pass

    def decode(self, ids, batch=False):
        decoded = self.tokenizer.batch_decode(ids) if batch else [
            self.tokenizer.decode(ids)]
        # remove spaces sentencepiece adds around newline
        for idx, sent in enumerate(decoded):
            decoded[idx] = sub(r"\s*\n\s*", "\n", sent)
        return decoded if batch else decoded[0]

    def train(self, **kwargs):
        if not self.args.overwrite_output_dir and isdir(output_dir := self.args.output_dir):
            last_checkpoint = get_last_checkpoint(output_dir)

            if isfile(join(output_dir, "config.json")):
                logger.info(
                    f"Output directory ({output_dir}) exists already and is not empty. Skipping training."
                )
                last_checkpoint = output_dir
            elif last_checkpoint:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `output_dir` or add `overwrite_output_dir` to train from scratch."
                )
        else:
            last_checkpoint = None

        kwargs["resume_from_checkpoint"] = last_checkpoint
        return super().train(**kwargs)

    def test(self, save_metrics=True):
        logger.info("Testing model.")
        ds = self.eval_dataset
        metrics = self.evaluate(eval_dataset=ds)

        metrics = {key.replace("eval", "test")
                               : value for key, value in metrics.items()}
        self.log_metrics('test', metrics)
        if save_metrics:
            self.save_metrics('test', metrics, False)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:

        if prediction_loss_only or self.args.prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "bad_words_ids": [[id_] for id_ in self.tokenizer.additional_special_tokens_ids],
            "max_length": self.args.max_length
            if self.args.max_length is not None
            else self.model.config.max_length,
            "num_beams": self.args.num_beams
            if self.args.num_beams is not None
            else self.model.config.num_beams,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask")
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask")

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if (
            hasattr(self.model, "encoder")
            and self.model.encoder.main_input_name != self.model.main_input_name
        ):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )

        if self.model.config.is_encoder_decoder:
            # add a start tensor so that eval code works (it expects a bos_token)
            start_tensor = full((len(generation_inputs), 1),
                                self.model.config.decoder_start_token_id, device=self.args.device)
            # and remove start-of-sequence token
            generated_tokens = cat(
                (start_tensor, generation_inputs, generated_tokens[:, 1:]), dim=1)
            pass

        # tensor([], device=self.args.device))
        return (None, generated_tokens, inputs['labels'])

    @cached_property
    def parameters(self):
        if hasattr(self, "model"):
            return sum(t.numel() for t in self.model.parameters())
        else:
            raise ValueError
