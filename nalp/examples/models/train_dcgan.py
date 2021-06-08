import tensorflow as tf

from nalp.datasets import ImageDataset
from nalp.models import DCGAN

# Loading the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Creating an Image Dataset
dataset = ImageDataset(x, batch_size=256, shape=(
    x.shape[0], 28, 28, 1), normalize=True)

# Creating the DCGAN
dcgan = DCGAN(input_shape=(28, 28, 1), noise_dim=100,
              n_samplings=3, alpha=0.3, dropout_rate=0.3)

# Compiling the DCGAN
dcgan.compile(d_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
              g_optimizer=tf.optimizers.Adam(learning_rate=0.0001))

# Fitting the DCGAN
dcgan.fit(dataset.batches, epochs=100)

# Saving DCGAN weights
dcgan.save_weights('trained/dcgan', save_format='tf')
