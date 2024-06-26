# Let the Poem Hit the Rhythm: Using a Byte-Based Transformer for Beat-Aligned Poetry Generation


## Model Summary

This repository contains an implementation of a beat-aligned poetry filler model using the ByT5 transformer.

This model allows users to insert a beat pattern into a poem or song lyric and generates candidates that align with the specified beat pattern while preserving the contextual meaning.

Think of this model as a tool for songwriting or poetry composition where you have gaps in your lines. Instead of leaving those gaps blank, you fill them with sounds mimicking the flow of words, similar to scat singing. For example: The "da-dam" is whispering something to me! Possible words that fit the flow could be: "heaven," "woman," or "city." This model assists in finding words that maintain the rhythmic flow.


## Encoding

A “1” represents a beat unit where there is a vowel onset, and a “0” represents a non-beat unit (or rest) where you have a consonant not followed by a vowel or a vowel not preceded by a consonant.

- `da-`: 1
- `-m`: 0
- `dam`: 10
- `da-da-dam-dam-da-dam`: 111010110


## Using Text2Text Pipeline

To get started with the model, use the following code. Note that the beat pattern should be enclosed between `<extra_id_0>` and `<extra_id_1>`.

```python
from transformers import pipeline

beat_aligned_generator = pipeline("text2text-generation", model='melzohbi/byt5-beat-align-base')
poem = "The <extra_id_0>110<extra_id_1> is whispering something to me. <extra_id_2>"
generated_words = beat_aligned_generator(poem, max_length=30, do_sample=True, num_return_sequences=5, temperature=0.8, top_p=1)
print(generated_words)

```


## Citation

If you use this model in your research, please cite the following paper:

```
@inproceedings{elzohbi2024let,
  title={Let the Poem Hit the Rhythm: Using a Byte-Based Transformer for Beat-Aligned Poetry Generation},
  author={Elzohbi, Mohamad and Zhao, Richard},
  booktitle={Proceedings of the 15th International Conference on Computational Creativity (ICCC)},
  year={2024}
}
```
