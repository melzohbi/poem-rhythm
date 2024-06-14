# This code has been adapted from the GitHub repository 'https://github.com/potamides/uniformers'
# Portions of the original code have been modified to fit the specific requirements
# of this project. Credit goes to the original authors for their contributions.

from uniformer_utils.utils import AbstractPoetryLMTrainer
from uniformer_utils.process_english import QuatrainV2Processing
from uniformer_utils.metrics import load_metric
import random
import wandb
from functools import partial
from random import randrange
import torch
from numpy import where
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from transformers.utils import logging
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
from dp.phonemizer import Phonemizer
import numpy as np
from datasets import load_from_disk
from uniformer_utils.datasets import load_dataset

logger = logging.get_logger("transformers")


def _add_special_tokens(tokenizer, texts):
    bos, eos = tokenizer.bos_token, tokenizer.eos_token
    return [bos + text + eos for text in texts]


# sample spans according to a geometric distribution
def sample_spans(tokens, budget_percentage, p, CorVs):
    # Total number of tokens
    n = len(tokens)

    # Calculate the total masking budget in terms of tokens
    budget = int(n * (budget_percentage / 100.0))

    # Keep track of indices to be masked
    masked_indices = set()

    # Sample a span length from a geometric distribution
    span_length = np.random.geometric(p)

    # Ensure the span length doesn't exceed the remaining budget
    span_length = min(span_length, budget)

    # Randomly select a starting point
    start_index = random.randint(0, n - 1)

    # Compute the span indices
    # Ensure span doesn't exceed sequence length
    end_index = min(start_index + span_length, n)

    # Add the indices to the masked set
    for i in range(start_index, end_index):
        masked_indices.add(i)

    # Create the subset of tokens based on the masked indices
    selected_tokens = [tokens[i] for i in sorted(masked_indices)]

    del tokens[start_index:start_index + span_length]

    chosen_words_corv = "".join(CorVs[start_index:start_index + span_length])
    tokens.insert(start_index, f"<extra_id_0>{chosen_words_corv}<extra_id_1>")

    tokens = " ".join(tokens)
    selected_tokens = " ".join(selected_tokens)

    return {
        "chosen_words": selected_tokens,
        "text_masked": tokens
    }

# Update the attention mask for ablation study


def update_attention_mask(input_ids, attention_mask):
    for i, seq in enumerate(input_ids):
        try:
            # Find start and end indices
            start_index = seq.index(259)+1
            end_index = seq.index(260)
            # Set values between start_index and end_index to 0
            if start_index < end_index:
                attention_mask[i][start_index:end_index +
                                  1] = [0] * (end_index - start_index + 1)
        except ValueError:
            # Handle the case where start_token_id or end_token_id are not in the list
            continue

    return attention_mask


def _tokenizer_v2(examples, tokenizer, is_encoder_decoder=False, multiple=False, ablation=False):
    inputs, labels = list(), list()

    for id, verse in enumerate(examples['clean_lines']):

        list_of_words = verse.split(" ")

        # select for one word
        try:
            if not multiple:
                chosen_word = random.choice(list_of_words)
                index = list_of_words.index(chosen_word)
                chosen_word_corv = examples['CorVs'][id].split(",")[index]

                list_of_words[index] = f"<extra_id_0>{chosen_word_corv}<extra_id_1>"
                text_masked = " ".join(list_of_words)
                chosen_word_text = f"{chosen_word}"
            else:
                # select for multiple words
                budget_percentage = 25
                p = 0.2
                result = sample_spans(
                    list_of_words, budget_percentage, p, examples['CorVs'][id].split(","))
                text_masked = result["text_masked"]
                chosen_word_text = result["chosen_words"]

        except:
            logger.info(
                f"Error with verse: {verse} and word: {chosen_word_text}")
            text_masked = "something <extra_id_0>CVCC<extra_id_1> wrong"
            chosen_word_text = "went"

        inputs.append(text_masked + "<extra_id_2>")
        labels.append(chosen_word_text)

    if is_encoder_decoder:
        model_inputs = tokenizer(inputs, add_special_tokens=False)

        # added to update the attention mask for ablation study
        if ablation:
            model_inputs["attention_mask"] = update_attention_mask(
                model_inputs["input_ids"], model_inputs["attention_mask"])

        labels = tokenizer(labels)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    else:
        return tokenizer(_add_special_tokens(tokenizer, [i + l for i, l in zip(inputs, labels)]))

# this function is similar to the above function, we can combine them into one function


def _tokenizer_v2_binary(examples, tokenizer, is_encoder_decoder=False, multiple=False, ablation=False):
    inputs, labels = list(), list()

    for id, verse in enumerate(examples['clean_lines']):
        list_of_words = verse.split(" ")

        # select for one word
        try:
            if not multiple:
                chosen_word = random.choice(list_of_words)
                index = list_of_words.index(chosen_word)
                chosen_word_corv = examples['binary'][id].split(",")[index]

                list_of_words[index] = f"<extra_id_0>{chosen_word_corv}<extra_id_1>"
                text_masked = " ".join(list_of_words)
                chosen_word_text = f"{chosen_word}"
            else:
                # select for multiple words
                budget_percentage = 25
                p = 0.2
                result = sample_spans(
                    list_of_words, budget_percentage, p, examples['binary'][id].split(","))
                text_masked = result["text_masked"]
                chosen_word_text = result["chosen_words"]

        except:
            logger.info(
                f"Error with verse: {verse} and word: {chosen_word_text}")
            text_masked = "something <extra_id_0>100<extra_id_1> wrong"
            chosen_word_text = "went"

        inputs.append(text_masked + "<extra_id_2>")
        labels.append(chosen_word_text)

    if is_encoder_decoder:
        model_inputs = tokenizer(inputs, add_special_tokens=False)

        # added to update the attention mask for ablation study
        if ablation:
            model_inputs["attention_mask"] = update_attention_mask(
                model_inputs["input_ids"], model_inputs["attention_mask"])

        labels = tokenizer(labels)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    else:
        return tokenizer(_add_special_tokens(tokenizer, [i + l for i, l in zip(inputs, labels)]))


# This is a custom trainer class for our purpose
class PoetryCVTrainer(AbstractPoetryLMTrainer):
    def __init__(
        self,
        model,
        lang="en",
        batch_size=128,
        test_run=False,
        low_resource=False,
        model_type="binary",
        multiple_words=False,
        ablation=False,
        **kwargs,
    ):

        super().__init__(
            model=model,
            batch_size=batch_size,
            test_run=test_run,
            eval_multiplier=5 if test_run else 75,
            **kwargs,
        )

        self.eval_table = wandb.Table(columns=["ex_id", "original word", "corv_original", "predicted word", "masked sentence",
                                      "corv_score", "lev_score", "t5-prp-fluency"])

        if model.config.is_encoder_decoder:
            data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        else:
            data_collator = DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # these can be arguments.
        lang_model = "papluca/xlm-roberta-base-language-detection"
        phonemizer_checkpoint = 'en_us_cmudict_ipa_forward.pt'

        logger.info(f"Loading language detection model {lang_model}")
        self.detection_tokenizer = AutoTokenizer.from_pretrained(
            lang_model)
        self.detection_model = XLMRobertaForSequenceClassification.from_pretrained(
            lang_model).to(self.device).eval()

        logger.info(f"Loading phonemizer model {phonemizer_checkpoint}")
        self.phonemizer = Phonemizer.from_checkpoint(phonemizer_checkpoint)

        train_data, eval_data = self.load_dataset(
            lang,
            batch_size,
            test_run,
            low_resource,
            model_type,
            multiple_words,
            ablation,
        )

        super(AbstractPoetryLMTrainer, self).__init__(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=data_collator,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=partial(
                self.compute_metrics,
                lang,
                batch_size,
                model_type,
            ),
            **self.trainer_args,
        )

    def save_state(self):
        super().save_state()
        if wandb.run:
            wandb.run.log({"eval_table": self.eval_table})

    def compute_metrics(self, lang, bs, model_type, p):
        labels = p.label_ids[0] if isinstance(
            p.label_ids, tuple) else p.label_ids
        labels = self.decode(
            where(labels == -100, self.tokenizer.pad_token_id, labels), batch=True)
        preds = super().compute_metrics(p)
        corv_string_list, preds_, masked_sentences_ = list(), list(), list()
        for idx, pred in enumerate(preds):
            tokenized = self.tokenizer.tokenize(pred)
            # mask_token = self.tokenizer.tokenize(" [MASK] ")
            corv_list = tokenized[tokenized.index(
                '<extra_id_0>')+1:tokenized.index('<extra_id_1>')]
            corv_string_list.append(
                self.tokenizer.convert_tokens_to_string(corv_list))
            preds[idx] = self.tokenizer.convert_tokens_to_string(
                tokenized[tokenized.index('<extra_id_2>')+1:])

            # adding these for the purpose of t5 loss evaluation
            preds_.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(
                "<extra_id_0> ") + tokenized[tokenized.index('<extra_id_2>')+1:]).replace("<pad>", ""))
            masked_sentence_ = tokenized[1:tokenized.index('<extra_id_0>')] + self.tokenizer.tokenize(
                " <extra_id_0> ") + tokenized[tokenized.index('<extra_id_1>')+1:tokenized.index('<extra_id_2>')]
            masked_sentences_.append(
                self.tokenizer.convert_tokens_to_string(masked_sentence_).replace("<pad>", ""))

        logger.info(
            f"Computing metrics with the convowel metric and the cohwordorig metrics")

        if model_type == "binary":
            convowel_score = load_metric("convowelencode", batch_size=bs, phonemizer=self.phonemizer).compute(
                predicted_words=preds, corv=corv_string_list)

        else:
            convowel_score = load_metric("convowel", language=lang, batch_size=bs, phonemizer=self.phonemizer).compute(
                predicted_words=preds, corv=corv_string_list)

        t5coh_score = load_metric("t5coh", batch_size=bs).compute(
            texts=masked_sentences_, predicted_words=preds_)

        for idx in random.sample(range(len(preds)), 100):
            self.eval_table.add_data(idx, labels[idx], corv_string_list[idx], preds_[idx], masked_sentences_[idx],
                                     convowel_score['corv_score'], convowel_score['lev_score'], t5coh_score['t5-prp-fluency'])

        return convowel_score | t5coh_score

    def patch_tokenizer(self):
        super().patch_tokenizer()
        if not self.tokenizer.additional_special_tokens:
            special = {
                "additional_special_tokens": [f"<extra_id_{idx}>" for idx in range(3)],
                'pad_token': '<pad>'
            }
            self.tokenizer.add_special_tokens(special)  # pyright: ignore
            self.model.resize_token_embeddings(len(self.tokenizer))

    def classify_language(self, sentence):
        # Tokenize the input sentence
        inputs = self.detection_tokenizer(
            sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)

        # Predict the language
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

        # Map the prediction to the language

        predicted_languages = [self.detection_model.config.id2label[prediction.item()]
                               for prediction in predictions]

        return [lang == 'en' for lang in predicted_languages]

    def load_dataset(self, lang, bs, test, low_res, model_type, multiple_words, ablation):

        dataset_class = "quatrainv2"
        logger.info(f"Loading dataset from class {dataset_class}")
        raw_dataset = load_dataset(dataset_class, lang=lang, split="train" + (
            "[:20000]" if test else ""))

        # load only english verses
        logger.info(
            f"Filtering dataset for english verses with a language model")
        english_dataset = raw_dataset.filter(
            lambda x: self.classify_language(x['line']), batched=True)  # slow

        logger.info(f"Processing dataset with phonemizer")
        dataset = english_dataset.map(
            QuatrainV2Processing(
                lang=lang,
                phonemizer=self.phonemizer,
                batch_size=bs,
            ),
            batched=True,
        )

        # save the dataset to disk to save time
        # dataset.save_to_disk("english_quatrainv_Apr22.hf")

        # load the dataset from disk to save time
        # dataset = load_from_disk("english_quatrainv_Apr22.hf")
        # dataset.cleanup_cache_files()

        if test:
            dataset = dataset.shuffle(seed=42).select(range(10000))

        # tokenizing the dataset.
        logger.info(f"Tokenizing dataset with _tokenizer_v2 or binary version")
        tokenized_dataset = dataset.map(
            _tokenizer_v2_binary if model_type == "binary" else _tokenizer_v2,
            batched=True,
            fn_kwargs={  # pyright: ignore
                "tokenizer": self.tokenizer,
                "is_encoder_decoder": self.model.config.is_encoder_decoder,
                "multiple": multiple_words,
                "ablation": ablation,
            },
            load_from_cache_file=False
        )

        if low_res:
            tokenized_dataset, eval_tokenized_dataset = tokenized_dataset.train_test_split(
                test_size=(0.001 if not test else 0.5)).values()
        else:
            tokenized_dataset, eval_tokenized_dataset = tokenized_dataset.train_test_split(
                test_size=(0.005 if not test else 0.5)).values()  # pyright: ignore

        index = randrange(len(tokenized_dataset))
        sample = tokenized_dataset[index]
        detokenized = self.decode(sample["input_ids"])
        logger.info(
            f"Input sample {index} of the training set: {sample['input_ids']}")
        logger.info(
            f"Input sample {index} of the training set (detokenized): {detokenized}")
        if "labels" in sample:  # pyright: ignore
            detokenized = self.decode(sample["labels"])
            logger.info(
                f"Label sample {index} of the training set: {sample['labels']}")
            logger.info(
                f"Label sample {index} of the training set (detokenized): {detokenized}")

        return tokenized_dataset, eval_tokenized_dataset
