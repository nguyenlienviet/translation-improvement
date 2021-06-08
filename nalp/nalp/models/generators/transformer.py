import tensorflow as tf
from nalp.models.layers.encoder_layer import EncoderLayer
from nalp.models.layers.decoder_layer import DecoderLayer
from nalp.models.layers import GumbelSoftmax
from nalp.utils.transformer_utils import positional_encoding
import nalp.utils.constants as c


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, tau=5):
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        # Defining a property to hold the Gumbel-Softmax temperature parameter
        self.tau = tau

        # Creates a Gumbel-Softmax layer
        self.gumbel = GumbelSoftmax(name='gumbel')

    @property
    def tau(self):
      """float: Gumbel-Softmax temperature parameter.

      """

      return self._tau

    @tau.setter
    def tau(self, tau):
      self._tau = tau

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        # Lastly, we apply the Gumbel-Softmax layer
        x_g, y_g = self.gumbel(final_output, self.tau)


        return final_output, x_g, y_g, attention_weights

    def generate_greedy_search(self, start, max_length=100):
        """Generates text by using greedy search, where the sampled
        token is always sampled according to the maximum probability.

        Args:
            start (str): The start string to generate the text.
            max_length (int): Maximum length of generated text.

        Returns:
            A list holding the generated text.

        """

        # Encoding the start string into tokens and expanding its first dimension
        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(max_length):
            # Predicts the current token and gathers its last timestep
            _, preds, _ = self(start_tokens)
            preds = preds[:, -1, :]

            # Samples a predicted token
            sampled_token = tf.argmax(preds, -1).numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims(sampled_token, 0)

            # Decodes the token and appends to the output list
            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            # Checks if sampled token is an end-of-sentence and breaks the loop
            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_temperature_sampling(self, start, max_length=100, temperature=1.0):
        """Generates text by using temperature sampling, where the sampled
        token is sampled according to a multinomial/categorical distribution.

        Args:
            start (str): The start string to generate the text.
            max_length (int): Length of generated text.
            temperature (float): A temperature value to sample the token.

        Returns:
            A list holding the generated text.

        """

        # Applying Gumbel-Softmax temperature as argument
        self.tau = temperature

        # Encoding the start string into tokens and expanding its first dimension
        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(max_length):
            # Predicts the current token and gathers its last timestep
            _, preds, _ = self(start_tokens)
            preds = preds[:, -1, :]

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a predicted token
            sampled_token = tf.argmax(preds, -1).numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims(sampled_token, 0)

            # Decodes the token and appends to the output list
            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            # Checks if sampled token is an end-of-sentence and breaks the loop
            if sampled_token == c.EOS:
                break

        return sampled_tokens

    def generate_top_sampling(self, start, max_length=100, k=0, p=0.0):
        """Generates text by using top-k and top-p sampling, where the sampled
        token is sampled according to the `k` most likely words distribution, as well
        as to the maximim cumulative probability `p`.

        Args:
            start (str): The start string to generate the text.
            max_length (int): Length of generated text.
            k (int): Indicates the amount of likely words.
            p (float): Maximum cumulative probability to be thresholded.

        Returns:
            A list holding the generated text.

        """

        # Encoding the start string into tokens and expanding its first dimension
        start_tokens = self.encoder.encode(start)
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for _ in range(max_length):
            # Predicts the current token and gathers its last timestep
            _, preds, _ = self(start_tokens)
            preds = preds[:, -1, :]

            # Checks if there is a provided `k`
            if k > 0:
                # Samples the top-k predictions and its indexes
                preds, preds_indexes = tf.math.top_k(preds, k)

            # If there is no provided `k`,
            # it means that we need to sort the predictions tensor
            else:
                # Gathers sorted predictions and its indexes
                preds, preds_indexes = tf.math.top_k(preds, preds.shape[-1])

            # Checks if there is a provided probability
            if p > 0.0:
                # Calculates the cumulative probability over the predictions' softmax
                cum_probs = tf.math.cumsum(tf.nn.softmax(preds), axis=-1)

                # Gathers a binary mask indicating whether indexes are below threshold
                ignored_indexes = cum_probs <= p

                # Also ensures that first index will always be true to prevent zero
                # tokens from being sampled
                ignored_indexes = tf.tensor_scatter_nd_update(ignored_indexes, [[0, 0]], [True])

                # Filters the predictions and its indexes
                preds = tf.expand_dims(preds[ignored_indexes], 0)
                preds_indexes = tf.expand_dims(preds_indexes[ignored_indexes], 0)

            # Samples the maximum top-k logit and gathers the real token index
            index = tf.argmax(preds, -1)[0]
            sampled_token = [preds_indexes[-1][index].numpy()]

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims(sampled_token, 0)

            # Decodes the token and appends to the output list
            sampled_token = self.encoder.decode(sampled_token)[0]
            sampled_tokens.append(sampled_token)

            # Checks if sampled token is an end-of-sentence and breaks the loop
            if sampled_token == c.EOS:
                break

        return sampled_tokens


if __name__ == "__main__":
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    sample_transformer = Transformer(
        num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000, rate=dropout_rate)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _, _, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)
