import tensorflow as tf

from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models import SeqGAN

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', corpus_type='word')

# Creating an IntegerEncoder, learning encoding and encoding tokens
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)

# Creating Language Modeling Dataset
dataset = LanguageModelingDataset(encoded_tokens, max_contiguous_pad_length=10, batch_size=4)

# Creating the SeqGAN
seqgan = SeqGAN(encoder=encoder, vocab_size=corpus.vocab_size, max_length=10, embedding_size=256,
                hidden_size=512, n_filters=(64, 128, 256), filters_size=(3, 5, 5), dropout_rate=0.25, temperature=1)

# Compiling the SeqGAN
seqgan.compile(pre_optimizer=tf.optimizers.Adam(learning_rate=0.01),
               d_optimizer=tf.optimizers.Adam(learning_rate=0.001),
               g_optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Pre-fitting the SeqGAN
seqgan.pre_fit(dataset.batches, g_epochs=50, d_epochs=10)

# Fitting the SeqGAN
seqgan.fit(dataset.batches, epochs=10, g_epochs=1, d_epochs=5, n_rollouts=16)

# Saving SeqGAN weights
seqgan.save_weights('trained/seqgan', save_format='tf')
