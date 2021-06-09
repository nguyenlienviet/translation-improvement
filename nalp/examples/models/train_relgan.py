import tensorflow as tf
import random, nltk
import numpy as np
import tensorflow_datasets as tfds

from nalp.corpus import TextCorpus
from nalp.models import RelGAN
from nalp.utils.transformer_utils import CustomSchedule
from nalp.utils.constants import BUFFER_SIZE

if __name__ == "__main__":
    model_name = "ted_hrlr_translate_pt_en_converter"
    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir='.', cache_subdir='', extract=True
    )
    tokenizers = tf.saved_model.load(model_name)

    # Creating a character TextCorpus from file
    corpus = TextCorpus(from_file='../../data/text/news_korean_to_english_google_pbmt.txt', corpus_type="word")

    sequences = corpus.sequences.copy()

    dataset = tf.data.Dataset.from_tensor_slices(sequences)

    def tokenize(en):
        en = tokenizers.en.tokenize(en)
        # Convert from ragged to dense, padding with zeros.
        en = en.to_tensor()
        return en

    BUFFER_SIZE = 20000
    BATCH_SIZE = 64

    def make_batches(ds):
        return (
            ds
                .cache()
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .map(tokenize, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))


    # TODO: train and validation split?
    train_batches = make_batches(dataset)

    good_corpus = TextCorpus(from_file='../../data/text/wiki_good_sentences.txt', corpus_type='word')

    encoded_good_tokens = tokenizers.en.tokenize(good_corpus.sequences)

    sequences = tf.data.Dataset.from_tensor_slices(encoded_good_tokens)

    # pad good sentences for disciminator
    sequences = tf.keras.preprocessing.sequence.pad_sequences(list(sequences.as_numpy_iterator()), dtype='int64',
                                                              padding='post', truncating='post', maxlen=7010)

    sequences = tf.data.Dataset.from_tensor_slices(sequences)

    good_datasets = sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    good_max_continuous_pad_length = list(good_datasets.take(1).as_numpy_iterator())[0].shape[1]

    # Create a dataset
    BUFFER_SIZE = 20000

    # Creating the RelGAN
    num_layers = 4
    d_model = 256
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    # TODO: check vocab size
    relgan = RelGAN(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                    input_vocab_size=tokenizers.en.get_vocab_size(), target_vocab_size=tokenizers.en.get_vocab_size(),
                    pe_input=1000, pe_target=1000, dropout_rate=dropout_rate,
                    max_length=good_max_continuous_pad_length - 1, n_filters=(64, 128, 256), filters_size=(3, 5, 5),
                    tau=5, tokenizers=tokenizers)

    # Compiling the GSGAN
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    relgan.compile(pre_optimizer=optimizer,
                   d_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                   g_optimizer=optimizer)

    # Pre-fitting the RelGAN
    relgan.pre_fit(train_batches=train_batches, epochs=20)

    # Fitting the RelGAN
    relgan.fit(batches=train_batches, good_batches=good_datasets, epochs=50)

    # Saving RelGAN weights
    relgan.save_weights('trained/relgan', save_format='tf')
