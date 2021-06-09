"""Relational Generative Adversarial Network.
"""

import tensorflow as tf
from tensorflow.keras.utils import Progbar

import nalp.utils.logging as l
from nalp.core import Adversarial
from nalp.models.discriminators import TextDiscriminator
from nalp.models.generators.transformer import Transformer
from nalp.utils.transformer_utils import create_masks, loss_function, accuracy_function

logger = l.get_logger(__name__)


class RelGAN(Adversarial):
    """A RelGAN class is the one in charge of Relational Generative Adversarial Networks implementation.

    References:
        W. Nie, N. Narodytska, A. Patel. Relgan: Relational generative adversarial networks for text generation.
        International Conference on Learning Representations (2018).

    """

    def __init__(self, num_layers=4, d_model=128, num_heads=8, dff=512, input_vocab_size=1, target_vocab_size=1,
                 pe_input=1000, pe_target=1000, max_length=1,
                 n_filters=(64), filters_size=(1), dropout_rate=0.1, tau=5, tokenizers=None):
        """Initialization method.

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder for the generator.
            vocab_size (int): The size of the vocabulary for both discriminator and generator.
            max_length (int): Maximum length of the sequences for the discriminator.
            embedding_size (int): The size of the embedding layer for both discriminator and generator.
            n_slots (int): Number of memory slots for the generator.
            n_heads (int): Number of attention heads for the generator.
            head_size (int): Size of each attention head for the generator.
            n_blocks (int): Number of feed-forward networks for the generator.
            n_layers (int): Amout of layers per feed-forward network for the generator.
            n_filters (tuple): Number of filters to be applied in the discriminator.
            filters_size (tuple): Size of filters to be applied in the discriminator.
            dropout_rate (float): Dropout activation rate.
            tau (float): Gumbel-Softmax temperature parameter.

        """

        logger.info('Overriding class: Adversarial -> RelGAN.')

        # Creating the discriminator network
        D = TextDiscriminator(max_length, d_model, n_filters, filters_size, dropout_rate)

        # Creating the generator network
        # G = GumbelRMCGenerator(encoder, vocab_size, embedding_size,
        #                        n_slots, n_heads, head_size, n_blocks, n_layers, tau)

        G = Transformer(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
            input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size,
            pe_input=pe_input, pe_target=pe_target, rate=dropout_rate)

        # Overrides its parent class with any custom arguments if needed
        super(RelGAN, self).__init__(D, G, name='RelGAN')

        self.tokenizers = tokenizers

        self.good_input_D = TextDiscriminator(max_length, d_model, n_filters, filters_size, dropout_rate)

        # Defining a property for holding the vocabulary size
        self.vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        # Gumbel-Softmax initial temperature
        self.init_tau = tau

        logger.info('Class overrided.')

    @property
    def vocab_size(self):
        """int: The size of the vocabulary.

        """

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self._vocab_size = vocab_size

    @property
    def init_tau(self):
        """float: Gumbel-Softmax initial temperature.

        """

        return self._init_tau

    @init_tau.setter
    def init_tau(self, init_tau):
        self._init_tau = init_tau

    def compile(self, pre_optimizer, d_optimizer, g_optimizer):
        """Main building method.

        Args:
            pre_optimizer (tf.keras.optimizers): An optimizer instance for pre-training the generator.
            d_optimizer (tf.keras.optimizers): An optimizer instance for the discriminator.
            g_optimizer (tf.keras.optimizers): An optimizer instance for the generator.

        """

        # Creates optimizers for pre-training, discriminator and generator
        self.P_optimizer = pre_optimizer
        self.D_optimizer = d_optimizer
        # TODO: maybe use different optimizer for transformer
        self.G_optimizer = g_optimizer

        # Defining the loss function
        # discriminator loss
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits

        # TODO: different loss for generator

        # Defining both loss metrics
        self.D_loss = tf.metrics.Mean(name='D_loss')
        self.G_loss = tf.metrics.Mean(name='G_loss')

        # Definining training accuracy metrics
        self.G_train_accuracy = tf.keras.metrics.Mean(name='train_G_accuracy')
        self.D_train_accuracy = tf.keras.metrics.Mean(name='train_D_accuracy')

        # Storing losses as history keys
        self.history['pre_G_loss'] = []
        self.history['pre_G_accuracy'] = []
        self.history['D_loss'] = []
        self.history['G_loss'] = []

    def generate_batch(self, x, good_batch):
        """Generates a batch of tokens by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            x (tf.tensor): A tensor containing the inputs.

        Returns:
            A (batch_size, length) tensor of generated tokens and a
            (batch_size, length, vocab_size) tensor of predictions.

        """

        # Gathers the batch size and maximum sequence length
        batch_size, max_length = x.shape[0], x.shape[1]

        # Creating an empty tensor for holding the Gumbel-Softmax predictions
        sampled_preds = tf.zeros([batch_size, 0, self.vocab_size])

        # Resetting the network states
        self.G.reset_states()

        tar_inp = good_batch[:, :-1]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, tar_inp)

        # TODO: Need to get masks !!
        _, preds, start_batch, _ = self.G(x, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)

        sampled_preds = tf.concat([sampled_preds, preds], 1)

        return start_batch, sampled_preds

    def _discriminator_loss(self, y_real, y_fake):
        """Calculates the loss out of the discriminator architecture.

        Args:
            y_real (tf.tensor): A tensor containing the real data targets.
            y_fake (tf.tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the discriminator network.

        """

        # Calculates the discriminator loss
        loss = self.loss(tf.ones_like(y_real), y_real - y_fake)

        return tf.reduce_mean(loss)

    def _generator_loss(self, y_real, y_fake):
        """Calculates the loss out of the generator architecture.

        Args:
            y_real (tf.tensor): A tensor containing the real data targets.
            y_fake (tf.tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the generator network.

        """

        # Calculating the generator loss
        loss = self.loss(tf.ones_like(y_fake), y_fake - y_real)

        return tf.reduce_mean(loss)

    train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64)]

    @tf.function(input_signature=train_step_signature)
    def G_pre_step(self, inp, tar):
        """Performs a single batch optimization pre-fitting step over the generator.

        Args:
            inp (tf.tensor): A tensor containing the bad input.
            tar (tf.tensor): A tensor containing the good input.

        """

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the logit-based predictions based on inputs
            predictions, _, _, _ = self.G(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)

            # Calculate the loss
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

            loss = loss_function(tar_real, predictions, loss_object)

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.G.trainable_variables)

        # Apply gradients using an optimizer
        self.P_optimizer.apply_gradients(
            zip(gradients, self.G.trainable_variables))

        # Updates the generator's accuracy state
        accuracy = accuracy_function(tar_real, predictions)
        self.G_train_accuracy.update_state(accuracy)

        # Updates the generator's loss state
        # Updates the generator's loss state
        self.G_loss.update_state(loss)

    # TODO: step needs to get good sentences data
    @tf.function(experimental_relax_shapes=True)
    def step(self, x, good_batch):
        """Performs a single batch optimization step.

        Args:
            x (tf.tensor): A tensor containing the inputs.
            y (tf.tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:

            # Generates new data, e.g., G(x)
            _, x_fake_probs = self.generate_batch(x, good_batch)

            # Samples fake targets from D(G(x))
            y_fake = self.D(x_fake_probs)

            # Extends the target tensor to an one-hot encoding representation
            # and samples real targets from D(x)
            good_batch = good_batch[:, 1:]
            y = tf.one_hot(good_batch, self.vocab_size)

            y_real = self.good_input_D(y)

            # Calculates both losses
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_fake, labels=tf.zeros_like(y_fake)))
            # G_loss = self._generator_loss(y_real, y_fake)
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_real, labels=tf.zeros_like(y_real)))
            D_loss = d_loss_real + d_loss_fake
            G_loss = tf.reduce_mean(-y_fake)
            # D_loss = self._discriminator_loss(y_real, y_fake)

        # Calculate both gradients
        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        # Applies both gradients using an optimizer
        self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(D_gradients, self.D.trainable_variables))

        # Updates both loss states
        self.G_loss.update_state(G_loss)
        self.D_loss.update_state(D_loss)

    def pre_fit(self, train_batches, epochs=100):
        """Pre-trains the model.

        Args:
            train_batches (Dataset): Pre-training batches containing samples.
            epochs (int): The maximum number of pre-training epochs.

        """

        logger.info('Pre-fitting generator ...')

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(train_batches).numpy()

        # Iterate through all generator epochs
        for e in range(epochs):
            logger.info('Epoch %d/%d', e + 1, epochs)

            # Resetting state to further append losses
            self.G_loss.reset_states()
            self.G_train_accuracy.reset_state()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(G)', 'accuracy(G)'])

            # Iterate through all possible pre-training batches
            for batch, tar in enumerate(train_batches):
                # Performs the optimization step over the generator
                self.G_pre_step(inp=tar, tar=tar)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(G)', self.G_loss.result()), ('accuracy(G)', self.G_train_accuracy.result())])

            # Dump loss to history
            self.history['pre_G_loss'].append(self.G_loss.result().numpy())
            self.history['pre_G_accuracy'].append(self.G_train_accuracy.result().numpy())

            logger.to_file('Loss(G): %s', self.G_loss.result().numpy())
            logger.to_file('accuracy(G): %s', self.G_train_accuracy.result().numpy())

    def fit(self, batches, good_batches, epochs=100):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing samples.
            epochs (int): The maximum number of training epochs.

        """

        logger.info('Fitting model ...')

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()
        print(n_batches)

        good_batches = list(good_batches.as_numpy_iterator())

        # Iterate through all epochs
        for e in range(epochs):
            logger.info('Epoch %d/%d', e + 1, epochs)

            # Resetting states to further append losses
            self.G_loss.reset_states()
            self.D_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(G)', 'loss(D)'])

            i = 0
            # Iterate through all possible training batches
            for batch, tar in enumerate(batches):
                # Performs the optimization step
                self.step(tar, good_batches[i])

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(G)', self.G_loss.result()), ('loss(D)', self.D_loss.result())])
                i += 1

            # Exponentially annealing the Gumbel-Softmax temperature
            self.G.tau = self.init_tau ** ((epochs - e) / epochs)

            # Dumps the losses to history
            self.history['G_loss'].append(self.G_loss.result().numpy())
            self.history['D_loss'].append(self.D_loss.result().numpy())

            logger.to_file('Loss(G): %s | Loss(D): %s', self.G_loss.result().numpy(), self.D_loss.result().numpy())