#
# FILENAME: main.py | Sample Transformer
# DESCRIPTION: Test code
# CREATED: 2024-01-01 @ 3:46 PM
# COPYRIGHT: Copyright (c) 2023-2024 by Ryan Smith <rysmith2113@gmail.com>
#

import pathlib, random, string

import tensorflow as tf

from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, TextVectorization
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import get_file
from keras.optimizers import Adam, SGD

from nltk.translate.bleu_score import sentence_bleu

import inputproc

from decoder_layer import DecoderLayer
from defaults import *
from embedding import Embedding
from encoder_layer import EncoderLayer
from positional_encoding import PositionalEncoder


def main() -> None:
    example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"
    
    inputs_en = [
        "I wonder what will come next.",
        "This is a basic example paragraph",
        "This has been split into its basic components"
    ]
    
    inputs_es = [
        "Me pregunto qué vendrá después.",
        "Este es un párrafo de ejemplo basico",
        "Este se ha dividido en sus componentes básicos."
    ]
    
    spacy_en, spacy_es = inputproc.load_tokenizers()
    sequences_en = inputproc.spacy_tokenize(inputs_en, spacy_en)
    sequences_es = inputproc.spacy_tokenize(inputs_es, spacy_es)
    
    text_file = get_file(
        fname="spa-eng.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
        extract=True,
    )
    text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"
    
    with open(text_file, encoding="utf8") as file:
        lines = file.read().split("\n")[:-1]
    
    en_to_es = []
    for line in lines:
        english, spanish = line.split("\t")
        spanish = "<bos> " + spanish + " <eos>"
        en_to_es.append((english, spanish))
    
    random.shuffle(en_to_es)
    num_val_samples = int(0.15 * len(en_to_es))
    num_train_samples = len(en_to_es) - 2 * num_val_samples
    train_pairs = en_to_es[:num_train_samples]
    val_pairs = en_to_es[num_train_samples:num_train_samples + num_val_samples]
    test_pairs = en_to_es[num_train_samples + num_val_samples:]
    
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    
    vocab_size = 15000
    sequence_length = 30

    source_vectorization = TextVectorization(
        max_tokens=vocab_size,
        split=inputproc.custom_split,
        output_mode="int",
        pad_to_max_tokens=True,
        output_sequence_length=sequence_length
    )
    
    target_vectorization = TextVectorization(
        max_tokens=vocab_size,
        split=inputproc.custom_split,
        output_mode="int",
        pad_to_max_tokens= True,
        output_sequence_length=sequence_length + 1 # Add one extra for Spanish punctuation
    )
    
    train_english_texts = [pair[0] for pair in train_pairs]
    train_spanish_texts = [pair[1] for pair in train_pairs]
    source_vectorization.adapt(train_english_texts)
    target_vectorization.adapt(train_spanish_texts)
    
    batch_size = 16
    
    def format_dataset(eng, spa):
        eng = source_vectorization(eng)
        spa = target_vectorization(spa)
        
        #spa_one_hot = tf.one_hot(spa, depth=vocab_size)
        
        return ({
            "english": eng,
            "spanish": spa[:, :-1],
        }, spa[:, 1:])

    def make_dataset(pairs):
        eng_texts, spa_texts = zip(*pairs)
        eng_texts = list(eng_texts)[:10000]
        spa_texts = list(spa_texts)[:10000]
        dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(format_dataset, num_parallel_calls=4)
        return dataset.shuffle(2048).prefetch(16).cache()
    
    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)
    
    for source, target in train_ds:
        print("Source:", source)
        print("Target:", target)
    
    #vocab_size = 1000
    d_model = 128
    d_ffn = d_model * 4
    num_heads = 2
    seq_len = 30
    
    src_vocab_size = source_vectorization.vocabulary_size()
    tgt_vocab_size = target_vectorization.vocabulary_size()
    print(src_vocab_size, tgt_vocab_size)
    
    # Encoder inputs
    encoder_inputs = Input(batch_size=batch_size, shape=(seq_len,), dtype=tf.dtypes.int64, name="english")
    encoder_embedding = Embedding(src_vocab_size, d_model, name="encoder_embedding")(encoder_inputs)
    encoder_inputs_with_pos = PositionalEncoder(d_model, DROPOUT_RATE, EXP_BASE, MAX_SEQUENCE_LEN, name="positional_encoder1")(encoder_embedding)
    encoder_outputs = EncoderLayer(d_model, d_ffn, num_heads, DROPOUT_RATE, EPSILON, dtype=tf.float64, name="encoder_layer")(encoder_inputs_with_pos)

    # Decoder inputs
    decoder_inputs = Input(batch_size=batch_size, shape=(seq_len,), dtype=tf.dtypes.int64, name="spanish")
    decoder_embedding = Embedding(tgt_vocab_size, d_model, name="decoder_embedding")(decoder_inputs)
    decoder_inputs_with_pos = PositionalEncoder(d_model, DROPOUT_RATE, EXP_BASE, MAX_SEQUENCE_LEN, name="positional_encoder2")(decoder_embedding)
    decoder_outputs, _ = DecoderLayer(d_model, d_ffn, tgt_vocab_size, num_heads, DROPOUT_RATE, EPSILON, dtype=tf.dtypes.float64, name="decoder_layer")(decoder_inputs_with_pos, encoder_outputs)

    logits = Dense(tgt_vocab_size, name="logits")(decoder_outputs)

    # Create a model
    transformer = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[logits], name="Transformer")

    # Print model summary
    transformer.summary()

    checkpoint = ModelCheckpoint(
        filepath=str(pathlib.Path("model/train1")),
        save_weights_only=False,  # Save the entire model, including architecture and optimizer state
        save_best_only=True,
        monitor="val_accuracy",
        mode="min",
        verbose=0
    )
    
    optimizer = Adam(learning_rate=LEARNING_RATE, clipvalue=1.0)
    loss_func = SparseCategoricalCrossentropy(from_logits=True)
    transformer.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=["accuracy"]
    )
    
    transformer.fit(
        train_ds, batch_size=batch_size, epochs=EPOCHS,
        callbacks=[checkpoint], validation_data=val_ds
    )
    
    loss, accuracy = transformer.evaluate(val_ds)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)
    
    spa_vocab = target_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    max_decoded_sentence_length = 20
    
    import numpy as np
    
    def decode_sequence(input_sentence):
        tokenized_input_sentence = source_vectorization([input_sentence])
        decoded_sentence = "[start]"
        for i in range(max_decoded_sentence_length):
            tokenized_target_sentence = target_vectorization(
                [decoded_sentence])[:, :-1]
            predictions = transformer(
                [tokenized_input_sentence, tokenized_target_sentence])
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = spa_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token
            if sampled_token == "[end]":
                break
        return decoded_sentence
    
    test_eng_texts = [pair[0] for pair in test_pairs]
    for _ in range(5):
        input_sentence = random.choice(test_eng_texts)
        print("-")
        print(input_sentence)
        print(decode_sequence(input_sentence))
    
    test_eng_texts = [pair[0] for pair in test_pairs]
    test_spa_texts = [pair[1] for pair in test_pairs]
    score = 0
    bleu  = 0
    for i in range(20):
        candidate = decode_sequence(test_eng_texts[i])
        reference = test_spa_texts[i].lower()
        print(candidate,reference)
        score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        bleu += score
        print(f"Score:{score}")
    print(f"\nBLEU score : {round(bleu,2)}/20")
    
    # sources, targets = next(iter(train_ds.take(1)))
    # english, spanish = sources["english"], sources["spanish"]
    
    # embedding_layer = Embedding(src_vocab_size, dim_model, dtype=tf.dtypes.float64)
    # embeddings = embedding_layer(english)
    
    # encoder_inputs = Input(shape=(batch_size,), dtype=tf.dtypes.float64, name="english")
    
    # x = PositionalEncoder(dim_model, DROPOUT_RATE, EXP_BASE, sequence_length)(embeddings)
    # encoder_outputs = Encoder(dim_model, dim_ffn, num_layers, num_heads, DROPOUT_RATE, EPSILON)(x)

    # decoder_inputs = Input(shape=(batch_size,), dtype=tf.dtypes.float64, name="spanish")
    # embeddings = embedding_layer(spanish)
    # x = PositionalEncoder(dim_model, DROPOUT_RATE, EXP_BASE, sequence_length)(embeddings)
    # x = Decoder(tgt_vocab_size, dim_model, dim_ffn, num_layers, num_heads, DROPOUT_RATE, EPSILON)(targets, encoder_outputs, inputproc.build_target_mask(targets), inputproc.build_source_mask(encoder_outputs))
    # transformer = Model([encoder_inputs, decoder_inputs], embedding_layer(x))
    
    # transformer = make_model(src_vocab_size, tgt_vocab_size, dim_model, dim_ffn, num_layers,
    #                          num_heads, DROPOUT_RATE, EPSILON, EXP_BASE, sequence_length)
    
    # optimizer = SGD(learning_rate=0.001)
    # loss_function = SparseCategoricalCrossentropy()
    # transformer.compile(optimizer, loss_function)
    
    # train_ds = train_ds.take(4).cache("dataset.txt")
    # size = len(list(train_ds))
    # EPOCHS = 4
    
    # print("\n\nStarting training")
    # print(f"Dataset size: {size}")
    # print(f"Training over {EPOCHS} {'epochs' if EPOCHS > 1 else 'epoch'}")

    # Training loop
    # for epoch in range(EPOCHS):
    #     epoch_loss = Mean()
    #     epoch_accuracy = SparseCategoricalAccuracy()
        
    #     print(f"Start Epoch {epoch + 1}")

    #     # Iterate through the dataset
    #     for i, (source, target) in enumerate(train_ds):
    #         print(f"\tPass {i + 1}/{size}")
            
    #         english, spanish = source["english"], source["spanish"]
            
    #         with tf.GradientTape() as tape:
    #             predictions = transformer((english, target))
    #             loss = loss_function(target, predictions)

    #         gradients = tape.gradient(loss, transformer.trainable_variables)
    #         optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    #         epoch_loss.update_state(loss)
    #         epoch_accuracy.update_state(target, predictions)

    #     # Print epoch statistics
    #     print(f"End Epoch {epoch + 1}, Loss: {epoch_loss.result()}, Accuracy: {epoch_accuracy.result()}")
    
    #transformer.save("transformer.h5")
    
    # saving model achitecture in json file
    # with open("transformer.json", "w") as json_file:
    #     json_file.write(transformer.to_json())
    
    # spa_vocab = target_vectorization.get_vocabulary()
    # spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    # max_decoded_sentence_length = 20
    
    # def decode_sequence(input_sentence):
    #     tokenized_input_sentence = source_vectorization([input_sentence])
    #     decoded_sentence = "[start]"
    #     for i in range(max_decoded_sentence_length):
    #         tokenized_target_sentence = target_vectorization(
    #             [decoded_sentence])[:, :-1]
    #         predictions = transformer(
    #             [tokenized_input_sentence, tokenized_target_sentence])
    #         sampled_token_index = np.argmax(predictions[0, i, :])
    #         sampled_token = spa_index_lookup[sampled_token_index]
    #         decoded_sentence += " " + sampled_token
    #         if sampled_token == "[end]":
    #             break
    #     return decoded_sentence

    # test_eng_texts = [pair[0] for pair in test_pairs]
    # for _ in range(5):
    #     input_sentence = random.choice(test_eng_texts)
    #     print("-")
    #     print(input_sentence)
    #     print(decode_sequence(input_sentence))
    
    # from keras.utils import plot_model
    # plot_model(transformer, to_file='transformer.png', show_shapes=True)
    # from IPython.display import Image
    # Image("transformer.png")
    
if __name__ == "__main__":
    import sys
    sys.exit(main())