import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):    
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch,dim))
        return z_mean + tf.exp(.5 * z_log_var) * epsilon

class SQLiVAE(keras.Model):
    def __init__(self,encoder,decoder,**kwargs):
        super(SQLiVAE,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self,data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)           
            reconstruction = self.decoder(z)
            print(reconstruction)
            reconstruction_loss = keras.losses.mean_squared_error(data,reconstruction)
            #reconstruction_loss = tf.reduce_sum(keras.losses.mean_squared_error(data,reconstruction), axis=1)
            kl_loss = -.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = (tf.reduce_sum(kl_loss,axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss,self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    
def build_encdec(input_dim=200, latent_dim=2, layer1=100, layer2=25):
    
    encoder_inputs = keras.Input(shape=(input_dim,))

    x = layers.Dense(layer1,activation='relu')(encoder_inputs)
    x = layers.Dense(layer2,activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean,z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean,z_log_var,z], name='encoder')

    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(layer2,activation='relu')(latent_inputs)
    x = layers.Dense(layer1,activation='relu')(x)
    decoder_outputs = layers.Dense(input_dim,activation='relu')(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

    return encoder,decoder  