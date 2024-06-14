# This code has been adapted from the GitHub repository 'https://github.com/potamides/uniformers'
# Portions of the original code have been modified to fit the specific requirements
# of this project. Credit goes to the original authors for their contributions.

import torch
from datasets import Features, Value
from datasets.info import MetricInfo
import datasets.metric
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer


class T5Coherence(datasets.metric.Metric):
    """Score coherence in a quatrain using T5 model"""

    def __init__(
        self,
        model_name="google-t5/t5-small",
        **kwargs,
    ):
        kwargs["config_name"] = model_name
        super().__init__(**kwargs)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def _info(self):
        return MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features({
                "predicted_words": Value("string"),
                "texts": Value("string")
            }),
        )

    def calculate_word_score(self, texts, masked_words, batch_size=16):
        num_batches = len(texts) // batch_size + \
            (1 if len(texts) % batch_size != 0 else 0)
        total_loss = 0

        for i in range(num_batches):
            # Select the batch of texts and masked words
            batch_texts = texts[i * batch_size: (i + 1) * batch_size]
            batch_masked_words = masked_words[i *
                                              batch_size: (i + 1) * batch_size]

            # Tokenize the batch
            tokenized_texts = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            )
            labels = self.tokenizer(
                batch_masked_words, return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():
                loss_score = self.model(
                    input_ids=tokenized_texts['input_ids'], labels=labels['input_ids']
                ).loss

            total_loss += loss_score.item()  # Convert tensor to a Python float for addition

        return total_loss / num_batches  # Return the average loss score for the batches

    def _compute(
        self,
        texts,
        predicted_words,

    ):
        perplexity = self.calculate_word_score(texts, predicted_words)
        return {
            "t5-prp-fluency": perplexity,
        }
