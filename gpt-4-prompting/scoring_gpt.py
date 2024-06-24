# This script is used to score the output of the GPT-4 model on the beat-aligned poetry generation task.
# It uses the phonemizer to convert the output to phonemes and then calculates the CV pattern of the phonemes.
# The CV pattern is then compared to the CV pattern of the real word and the similarity is calculated using the Levenshtein distance.
# The final score is the average of the similarity scores of all the words in the dataset.

import pandas as pd
from dp.phonemizer import Phonemizer
from Levenshtein import distance as levenshtein_distance
from statistics import mean
from uniformer_utils.process_english import get_corv, encode_cv_binary

if __name__ == '__main__':
    phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa_forward.pt')

    df = pd.read_csv("gpt-4-results/compiled_results_for_testing.csv")

    # apply phonemizer to the prediction column and save the result in a new column, the phonemizer takes an argument lang='en_us'
    df['phonemes'] = df['prediction'].apply(
        lambda x: phonemizer(x, lang='en_us', batch_size=1))
    df['real_corv'] = df['phonemes'].apply(lambda x: get_corv(x))
    df['real_pattern'] = df['real_corv'].apply(lambda x: encode_cv_binary(x))

    scores = list()
    lev_distances = list()

    pattern_values = df['real_pattern'].values
    predicted_words = df['recon_pattern'].values.astype(str)

    # df.to_csv("gpt-4-results/results.csv", index=False)

    for i in range(0, len(pattern_values)):
        # exact accuracy
        scores.append(float(pattern_values[i] == predicted_words[i]))

        # levenshtein distance
        distance = levenshtein_distance(
            pattern_values[i], predicted_words[i])
        # convert distance to a score between 0 and 1 similarity (max score must be 0)
        score = 1 - min(distance / (len(pattern_values[i]) + 0.0005), 1)

        lev_distances.append(score)

    print(mean(scores))
    print(mean(lev_distances))
