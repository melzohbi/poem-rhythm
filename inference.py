
from transformers import pipeline

if __name__ == "__main__":

    beat_aligned_generator = pipeline(
        "text2text-generation", model='melzohbi/byt5-beat-align-base')
    poem = "The <extra_id_0>110<extra_id_1> is whispering something to me. <extra_id_2>"
    generated_words = beat_aligned_generator(poem, max_length=30, do_sample=True,
                                             num_return_sequences=5, temperature=0.7, top_p=1)

    print(generated_words)
