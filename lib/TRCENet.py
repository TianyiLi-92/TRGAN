import numpy as np
import tensorflow as tf
from tensorflow import keras

#import lib.model as model
import lib.model_Michele as model
import lib.ops as ops

# Definition of the generator
class TRCENet(keras.Model):
    def __init__(self, FLAGS):
        super(TRCENet, self).__init__()
        self.FLAGS = FLAGS

        #self.netE = model.encoder(FLAGS.nei, FLAGS.nc, FLAGS.kW, FLAGS.sW, FLAGS.nef, FLAGS.nBottleneck)
        #self.netG = model.decoder(FLAGS.nBottleneck, FLAGS.noiseGen, FLAGS.nz, FLAGS.ngf, FLAGS.nc, FLAGS.ngo)
        self.netE = model.encoder_Michele(FLAGS.nei, FLAGS.nc, FLAGS.ekW, FLAGS.esW)
        self.netG = model.decoder_Michele(FLAGS.nc, FLAGS.dkW, FLAGS.dsW, FLAGS.ngo)

        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        self.vel_grad_loss_tracker = keras.metrics.Mean(name="vel_grad_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")

        self.epoch_step = tf.Variable(0, trainable=False)

    def call(self, inputs):
        # Build the generator part
        latent_vector = self.netE(inputs)

        if self.FLAGS.noiseGen:
            noise_shape = tf.concat([tf.shape(inputs)[0:1], tf.constant([1, 1, self.FLAGS.nz])], 0)

            if self.FLAGS.noisetype == 'uniform':
                noise = tf.random.uniform(shape=noise_shape, minval=-1, maxval=1, seed=1)
            elif self.FLAGS.noisetype == 'normal':
                noise = tf.random.normal(shape=noise_shape, seed=1)

            return self.netG([latent_vector, noise])
        else:
            return self.netG(latent_vector)

    def initialize(self):
        if self.FLAGS.checkpoint is not None:
            print("Restoring weights from {}".format(self.FLAGS.checkpoint))
            self.load_weights(self.FLAGS.checkpoint)

# =============================================================================
#     def velocity_grad_loss(self, context_train, gap_train, gap_train_pred):
#         dg_dx = ops.tf_ddx_fft(context_train, gap_train, self.FLAGS.mask)
#         dg_dx_pred = ops.tf_ddx_fft(context_train, gap_train_pred, self.FLAGS.mask)
#         dg_dx_rms = tf.math.reduce_mean( tf.math.square(dg_dx), axis=[1, 2, 3], keepdims=True )
#         ddx_loss = tf.math.reduce_mean( tf.math.square(dg_dx_pred - dg_dx) / dg_dx_rms )
# 
#         dg_dy = ops.tf_ddy_fft(context_train, gap_train, self.FLAGS.mask)
#         dg_dy_pred = ops.tf_ddy_fft(context_train, gap_train, self.FLAGS.mask)
#         dg_dy_rms = tf.math.reduce_mean( tf.math.square(dg_dy), axis=[1, 2, 3], keepdims=True )
#         ddy_loss = tf.math.reduce_mean( tf.math.square(dg_dy_pred - dg_dy) / dg_dy_rms )
# 
#         vel_grad_loss = ddx_loss + ddy_loss
# 
#         return vel_grad_loss
# =============================================================================

    def train_step(self, data):
        # Unpack the data.
        context_train, gap_train = data

        with tf.GradientTape() as tape:
            gap_train_pred = self(context_train, training=True)  # Forward pass
            # Compute our own losses
            # mse loss
            #wtl2Matrix = ops.tf_wtl2Matrix(gap_train, self.FLAGS.overlapPred)
            #gap_train_rms = tf.math.reduce_mean( wtl2Matrix * tf.math.square(gap_train), axis=[1, 2, 3], keepdims=True )
            #mse_loss = tf.math.reduce_mean( wtl2Matrix * tf.math.square(gap_train_pred - gap_train) / gap_train_rms )
            mse_loss = tf.math.reduce_mean( tf.math.square(gap_train_pred - gap_train) )
            # velocity gradient loss
            vel_grad_loss = 0#self.velocity_grad_loss(context_train, gap_train, gap_train_pred)
            # Content loss => mse + velocity gradient
            content_loss = (1 - self.FLAGS.lambda_vel_grad) * mse_loss + self.FLAGS.lambda_vel_grad * vel_grad_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(content_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.mse_loss_tracker.update_state(mse_loss)
        self.vel_grad_loss_tracker.update_state(vel_grad_loss)
        self.content_loss_tracker.update_state(content_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.mse_loss_tracker, self.vel_grad_loss_tracker, self.content_loss_tracker]

    def test_step(self, data):
        # Unpack the data.
        context_train, gap_train = data
        # Compute predictions
        gap_train_pred = self(context_train, training=False)
        # Updates the metrics tracking the loss
        # mse loss
        #wtl2Matrix = ops.tf_wtl2Matrix(gap_train, self.FLAGS.overlapPred)
        #gap_train_rms = tf.math.reduce_mean( wtl2Matrix * tf.math.square(gap_train), axis=[1, 2, 3], keepdims=True )
        #mse_loss = tf.math.reduce_mean( wtl2Matrix * tf.math.square(gap_train_pred - gap_train) / gap_train_rms )
        mse_loss = tf.math.reduce_mean( tf.math.square(gap_train_pred - gap_train) )
        self.mse_loss_tracker.update_state(mse_loss)
        # velocity gradient loss
        vel_grad_loss = 0#self.velocity_grad_loss(context_train, gap_train, gap_train_pred)
        self.vel_grad_loss_tracker.update_state(vel_grad_loss)
        # Content loss => mse + velocity gradient
        content_loss = (1 - self.FLAGS.lambda_vel_grad) * mse_loss + self.FLAGS.lambda_vel_grad * vel_grad_loss
        self.content_loss_tracker.update_state(content_loss)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


    #debug code::::::::::::::::::::::::::::::::::======================================

    # def train_step(self, data):
    #     # Unpack the data.
    #     context_train, gap_train = data

    #     with tf.GradientTape() as tape:
    #         gap_train_pred = self(context_train, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in 'compile()')
    #         loss = self.compiled_loss(gap_train, gap_train_pred, regularization_losses=self.losses)

    #         loss22 = keras.losses.mean_squared_error(gap_train, gap_train_pred)

    #     tf.print('losssssssssssssssssss=', loss)
    #     tf.print('losssssssssssssssssss22=', tf.reduce_mean(loss22))
    #     for m in self.metrics:
    #         tf.print(m.name, 'results=', m.result())

    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)

    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #     # Update metrics (includes the metric that tracks the loss)
    #     #self.compiled_metrics.update_state(gap_train, gap_train_pred)

    #     self.loss_tracker.update_state(loss22)

    #     # Return a dict mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}

    # def test_step(self, data):
    #     # Unpack the data.
    #     context_train, gap_train = data
    #     # Compute predictions
    #     gap_train_pred = self(context_train, training=False)

    #     loss22 = keras.losses.mean_squared_error(gap_train, gap_train_pred)

    #     # Updates the metrics tracking the loss
    #     loss1 = self.compiled_loss(gap_train, gap_train_pred, regularization_losses=self.losses)

    #     tf.print('ssssssolll', loss1)
    #     tf.print('ssssssolll22', tf.reduce_mean(loss22))

    #     for m in self.metrics:
    #         tf.print(m.name, 'val_results=', m.result())

    #     #self.loss_tracker.update_state(loss22)

    #     # Return a dict mapping metric names to current value.
    #     # Note that it will include the loss (tracked in self.metrics).
    #     return {m.name: m.result() for m in self.metrics}