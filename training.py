# This code has been adapted from the GitHub repository 'https://github.com/potamides/uniformers'
# Portions of the original code have been modified to fit the specific requirements
# of this project. Credit goes to the original authors for their contributions.

from trainers import PoetryCVTrainer
import wandb
from argparse import ArgumentParser

from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer_utils import set_seed
from transformers.utils.logging import (
    enable_explicit_format,
    get_logger,
    set_verbosity_debug,
    set_verbosity_info,
    add_handler
)
import logging

import os
from os.path import join
from os.path import basename
from functools import partial

# set up logging
set_verbosity_info()
enable_explicit_format()
logger = get_logger("transformers")

# set seed
set_seed(0)

# set up logger


def setup_logger(args):

    log_file_format = "[%(lineno)d]%(asctime)s: %(message)s"
    log_console_format = "%(message)s"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    console_handler = logging.FileHandler(
        os.path.join(args.out_dir, "info_logs.log"))
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_console_format))

    file_handler = logging.FileHandler(
        os.path.join(args.out_dir, "debug_logs.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_file_format))

    add_handler(console_handler)
    add_handler(file_handler)


if __name__ == "__main__":
    argument_parser = ArgumentParser(
        description="Fine-tune ByT5 for beat aligned poetry generation"
    )
    argument_parser.add_argument(
        "--model_name_or_path",
        default="google/byt5-small",
        help="name of the base model in huggingface hub or path if local",
    )
    argument_parser.add_argument(
        "--out_dir",
        default="models",
        help="directory where to write the model files",
    )
    argument_parser.add_argument(
        "--grad_acc_steps",
        default=8,
        type=int,
        help="number of gradient accumulation steps",
    )
    argument_parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="number of samples per batch",
    )
    argument_parser.add_argument(
        "--debug",
        action="store_true",
        help="perform a test run on debug verbosity",
    )

    argument_parser.add_argument(
        "--type",
        default="binary",
        help="models type, beat binary sequence or CorV",
    )

    argument_parser.add_argument(
        "--multiple_words",
        action="store_true",
        help="Should we consider masking multiple words or only single word",
    )

    argument_parser.add_argument(
        "--ablation",
        action="store_true",
        help="perform ablation study",
    )

    argument_parser.add_argument(
        "--description",
        help="description of the experiment",
    )

    args = argument_parser.parse_args()

    if args.debug:
        args.out_dir = join(args.out_dir, "debug")

    if args.debug:
        set_verbosity_debug()

    setup_logger(args)

    # write all arguments to log file
    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    # Initialize wandb
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="poetry_CV_training",
        # Track hyperparameters and run metadata
        config=args.__dict__,
    )

    # Set up trainer
    Trainer = partial(
        PoetryCVTrainer,
        output_dir=join(args.out_dir, basename(args.model_name_or_path)),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        gradient_checkpointing=False,
        test_run=args.debug,
        model_type=args.type,
        multiple_words=args.multiple_words,
        ablation=args.ablation
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trainer = Trainer(model=model, tokenizer=tokenizer)

    # train
    trainer.train()

    # save model
    trainer.save_model()
    trainer.save_state()
    model = trainer.model

    # test
    trainer.test()
