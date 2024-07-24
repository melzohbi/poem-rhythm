# This code has been adapted from the GitHub repository 'https://github.com/potamides/uniformers'
# Portions of the original code have been modified to fit the specific requirements
# of this project. Credit goes to the original authors for their contributions.

import datasets.metric
from datasets import Features, Value
from datasets.info import MetricInfo
from statistics import mean
from uniformer_utils.process_english import get_corv, encode_cv_binary
from transformers.utils import logging
import random
# from Levenshtein import distance as levenshtein_distance
import rapidfuzz.distance.Levenshtein as _Levenshtein

logger = logging.get_logger("transformers")


class Convowelencode(datasets.metric.Metric):
    """Score beat alignment in the masked words of a quatrain"""

    def __init__(
        self,
        batch_size=1,
        phonemizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.phonemizer = phonemizer

    def _info(self):
        return MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(
                {
                    "predicted_words": Value("string"),
                    "corv": Value("string"),
                }
            ),
        )

    def _preprocess(self, predicted_words):
        processed_predicted_words = list()
        for predicted_word in predicted_words:
            predicted_word = predicted_word.replace("<pad>", "")

            # added this line to remove the extra spaces and allow the prediction of multiple words
            predicted_word = predicted_word.split(" ")

            # remove empty strings
            predicted_word = [
                single_word for single_word in predicted_word if single_word]

            phomized_words = self.phonemizer(
                predicted_word, lang='en_us', batch_size=self.batch_size)

            # get the corv pattern of each word
            corv_predicted = [encode_cv_binary(get_corv(single_word)) if single_word and encode_cv_binary(get_corv(
                single_word)) else "None" for single_word in phomized_words]

            if "None" not in corv_predicted:
                corv_predicted = "".join(corv_predicted)
            else:
                corv_predicted = "None"

            processed_predicted_words.append(corv_predicted)

        return processed_predicted_words

    def _compute(
        self,
        predicted_words,
        corv,
    ):
        predicted_words = self._preprocess(predicted_words)
        scores = list()
        lev_distances = list()

        for i in range(0, len(predicted_words)):
            scores.append(float(predicted_words[i] == corv[i]))

            # use this similarity instead, it takes into consideration the length of both strings
            score = _Levenshtein.normalized_similarity(
                predicted_words[i], corv[i])

            # distance = levenshtein_distance(
            #     predicted_words[i], corv[i])
            # # convert distance to a score between 0 and 1 similarity
            # score = 1 - distance / (len(predicted_words[i]) + 0.0005)

            lev_distances.append(score)

        # get a random number between 0 and len(predicted_words)
        i = random.randint(0, len(predicted_words) - 1)
        logger.info(
            f"Sample: predicted word pattern is {predicted_words[i]} and original pattern is {corv[i]}")

        output_dict = {
            "corv_score": mean(scores),
            "lev_score": mean(lev_distances)
        }
        return output_dict
