# a class that process the quatrain dataset and return only the quatrain line that is in english

from gruut_ipa import Pronunciation
import re


def clean_lines(lines):
    regex_patterns = {
        r"hYpppHeN": "-",
        r'-': " ",
        r"\s+\'(s|ll|re|t|ve|m|er|d)(\s|$|,|.|;)": r"'\1 ",
        r"(?<![a-zA-Z])'(?![a-zA-Z])|[^a-zA-Z\s']": "",
        r"\s*n\'t ": "n't ",
        r"\s+": " ",
        r"^\s+|\s+$": ""
    }

    for pattern, replacement in regex_patterns.items():
        lines = [re.sub(pattern, replacement, line) for line in lines]

    return lines


def process(examples, phonemizer, bs):
    clean = clean_lines(examples["line"])

    phoneme_string_batch = phonemizer(
        clean, lang='en_us', batch_size=bs)

    if "phonemes" not in examples:
        examples["phonemes"] = [None] * len(phoneme_string_batch)
    if "CorVs" not in examples:
        examples["CorVs"] = [None] * len(phoneme_string_batch)

    if "binary" not in examples:
        examples["binary"] = [None] * len(phoneme_string_batch)

    if "clean_lines" not in examples:
        examples["clean_lines"] = [None] * len(clean)

    for idx, phoneme_string in enumerate(phoneme_string_batch):
        phoneme_string_list = phoneme_string.split()
        examples["phonemes"][idx] = ",".join(phoneme_string_list)
        examples["CorVs"][idx] = ",".join(corv(phoneme_string_list))
        examples["binary"][idx] = ",".join(
            (b_encoding(phoneme_string_list)))

        examples["clean_lines"][idx] = clean[idx]
    return examples


def corv(phoneme_string_list):
    corv = list()
    for word_phoneme in phoneme_string_list:
        corv.append(get_corv(word_phoneme))
    return corv


def b_encoding(phoneme_string_list):
    corv = list()
    for word_phoneme in phoneme_string_list:
        corv.append(encode_cv_binary(get_corv(word_phoneme)))
    return corv


def encode_cv_binary(input_string):
    output_string = ""
    i = 0
    while i < len(input_string):
        if input_string[i] == "C" or input_string[i] == "S":
            if i + 1 < len(input_string) and input_string[i + 1] == "V":
                output_string += "1"
                i += 2
            else:
                output_string += "0"
                i += 1
        elif input_string[i] == "V":
            if i == 0 or input_string[i - 1] != "C":
                output_string += "0"
                i += 1
            else:
                i += 1
        else:
            i += 1
    return output_string


def get_corv(phonemes):
    phonemes = Pronunciation.from_string(phonemes.replace("-", ""))
    corv_string = ""
    for idx, phone in enumerate(phonemes):
        if phone.is_vowel or "ɝ" in str(phone):
            if idx == 0:
                corv_string += "S"
            if len(phone.letters) == 2 or phone.is_long:
                corv_string += "VV"
            else:
                corv_string += "V"
        elif phone.is_consonant:
            corv_string += "C"
        elif "ː" in str(phone):
            corv_string += "V"
        else:
            # print(phone)
            corv_string += "X"
    return corv_string


class QuatrainV2Processing:
    def __init__(self, lang, phonemizer, batch_size=1):
        self.lang = lang
        self.bs = batch_size
        self.phonemizer = phonemizer

    def __call__(self, examples):
        examples = process(examples, self.phonemizer, self.bs)
        return examples
