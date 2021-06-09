import tensorflow as tf

from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models.generators import BiLSTMGenerator

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', corpus_type='char')

# Creating an IntegerEncoder, learning encoding and encoding tokens
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)

# Creating Language Modeling Dataset
dataset = LanguageModelingDataset(encoded_tokens, max_contiguous_pad_length=10, batch_size=64)

# Creating the LSTM
lstm = BiLSTMGenerator(encoder=encoder, vocab_size=corpus.vocab_size, embedding_size=256, hidden_size=512)

# As NALP's BiLSTMs are stateful, we need to build it with a fixed batch size
lstm.build((64, None))

# Compiling the LSTM
lstm.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Fitting the LSTM
lstm.fit(dataset.batches, epochs=100)

# Evaluating the LSTM
# lstm.evaluate(dataset.batches)

# Saving LSTM weights
lstm.save_weights('trained/bi_lstm', save_format='tf')
