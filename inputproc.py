#
# FILENAME: inputproc.py | Sample Transformer
# DESCRIPTION: Process text input to be sent to the transformer
# CREATED: 2023-12-31 @ 11:20 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

import os, re, string

import keras.utils
import spacy
import tensorflow as tf

from nltk import download
from textblob.tokenizers import WordTokenizer

###################################################
#
# Code to download the TextBlob required corpora
#

class __NLTKDownloadError(Exception):
    """ NTLK failed to download a package. """
    
    def __init__(self, message: str) -> None:
        super().__init__(message)

def download_corpora() -> None:
    TEXTBLOB_CORPORA = ["brown", "punkt", "wordnet", "averaged_perceptron_tagger", "conll2000", "movie_reviews"]

    for package in TEXTBLOB_CORPORA:
        if not download(package):
            raise __NLTKDownloadError(f"Package \"{package}\" failed to download")
    print("\nAll packages downloaded successfully.")

#
#
#
###################################################

def textblob_tokenize(sentence: str, with_special_tokens: bool = True) -> list[str]:
    """
        Tokenize a sentence using TextBlob's `WordTokenizer`. The tokenized output is a list of lowercase
        words with punctuation included. The resulting tokenized sequence also contains the beginning and
        end of sequence tokens, if specified.
    """
    
    tokens = WordTokenizer().tokenize(sentence.lower())
    return ["<bos>"] + tokens + ["<eos>"] if with_special_tokens else tokens

### SpaCy

def load_tokenizers() -> tuple[spacy.Language, ...]:
    """ Load English and Spanish tokenizers for training. """
    
    ENGLISH, SPANISH = "en_core_web_trf", "es_dep_news_trf"
    
    try:
        spacy_en = spacy.load(ENGLISH)
    except OSError:
        os.system(f"python -m spacy download {ENGLISH}")
        spacy_en = spacy.load(ENGLISH)
    
    try:
        spacy_es = spacy.load(SPANISH)
    except OSError:
        os.system(f"python -m spacy download {SPANISH}")
        spacy_es = spacy.load(SPANISH)
    
    print("Loaded English and Spanish tokenizers.")
    return spacy_es, spacy_en

def spacy_tokenize(texts: list[str], tokenizer: spacy.Language) -> list[list[int]]:
    """ Tokenize input sentences using a SpaCy tokenizer. """
    
    tokens = []
    
    docs = tokenizer.pipe(texts, batch_size=len(texts))
    for doc in docs:
        tokens.append([token.text.lower() for token in tokenizer(doc)])
    
    return tokens

def custom_split(text: str) -> list[str]:
    text = tf.strings.regex_replace(text, f"([{string.punctuation}])", r" \1 ")
    return tf.strings.split(text)

def custom_standardization(input_string: str, strip_chars: str) -> tf.Tensor:
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

def build_vocab(sentence: str) -> dict[str, int]:
    """
        Generate a vocabulary from an input. All tokens in the vocabulary are sorted alphabetically
        and mapped to an integer in the range `[0, sequence_length)`. The resulting vocabulary
        contains no duplicate tokens. The special tokens `<pad>`, `<bos>`, and `<eos>` are
        automatically inserted at the beginning of the vocabulary and always take the indices
        0, 1, and 2, respectively.
    """
    
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
    
    tokens = textblob_tokenize(sentence, with_special_tokens=False)
    tokens = SPECIAL_TOKENS + sorted(set(tokens))
    
    return {token: int_rep for int_rep, token in enumerate(tokens)}

def build_sequence(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """ Using a specific vocabulary, create a list of integers from a list of tokens. """
    return [vocab[token] for token in tokens]

def input_to_seq(input: str, vocab: dict[str, int]) -> list[int]:
    """
        Generate a sequence of integers from an input string. This is done as follows:	
           
        1. Tokenize the input. All tokenized output is a list of lowercase words with punctuation included.
        2. Use the vocabulary to create a sequence of integers from the list of tokens
        3. Return the created sequence
    """
    
    tokens = textblob_tokenize(input)
    return build_sequence(tokens, vocab)

def inputs_to_seqs(inputs: list[str], vocab: dict[str, int]) -> tf.Tensor:
    """ Convert a collection of input sentences to sequences. For more info, see `input_to_seq`. """
    
    #
    # TODO: Pad sequences with less tokens.
    #
    # Maybe set a max token limit and just pad up to that point? Or maybe just pad up to the token
    # length of the longest sequence.
    #
    
    sequences = [input_to_seq(sentence, vocab) for sentence in inputs]
    return pad_sequences(sequences)

def pad_sequences(sequences: list[list[int]], value: int = 0) -> tf.Tensor:
    """ Pad a list of tokenized sequences to a uniform length. """
    return tf.convert_to_tensor(keras.utils.pad_sequences(sequences, padding="post", value=value), dtype=tf.dtypes.int64)

def build_source_mask(source: tf.Tensor, value: int = 0) -> tf.Tensor:
    """ Generate a mask for a padded sequence of tokens. """
    
    #
    # Generate the source mask tensor. At each value in source, if the value doesn't equal the padding
    # value, the mask contains True, or 1. If the value does equal the padding value, the mask contains
    # False, or zero. The resulting mask has shape (batch_size, seq_len), which is the same as the input
    # tensor.
    #
    
    mask = tf.cast(source != value, dtype=tf.dtypes.int64)
    
    batch_size, seq_len = mask.get_shape()
    
    # Reshape the mask tensor to (batch_size, 1, 1, seq_len) to broadcast across a tensor of shape
    # (batch_size, n_heads, seq_len, seq_len)
    return tf.reshape(mask, [batch_size, tf.newaxis, tf.newaxis, seq_len])

def build_target_mask(target: tf.Tensor, value: int = 0) -> tf.Tensor:
    """ Generate a mask for a padded sequence of tokens. """
    
    #
    # Generate the target mask tensor. At each value in source, if the value doesn't equal the padding
    # value, the mask contains True, or 1. If the value does equal the padding value, the mask contains
    # False, or zero. The resulting mask has shape (batch_size, seq_len), which is the same as the input
    # tensor.
    #
    mask = tf.cast(target != value, dtype=tf.dtypes.int64)
    
    batch_size, seq_len = target.get_shape()
    
    # Create a lower triangle ones tensor to be used as a subsequent mask
    submask = tf.linalg.band_part(tf.ones([seq_len, seq_len], dtype=tf.dtypes.int64), -1, 0)
    
    # Reshape the submask to have the same shape as the original mask
    submask = tf.reshape(submask, [batch_size, tf.newaxis, seq_len, seq_len])
    
    # Use the submask to apply a bitmask to every value in the original mask. Reshape the mask tensor to
    # (batch_size, 1, seq_len, seq_len) to broadcast across a tensor of shape (batch_size, n_heads, seq_len, seq_len).
    target_mask = mask[:, tf.newaxis, :] & submask
    
    return target_mask