import tensorflow as tf
from tensorflow import keras

#import lib.model as model
import lib.model_Michele as model
import lib.ops as ops

class TRGAN(keras.Model):
    def __init__(self, FLAGS):
        super(TRGAN, self).__init__()
        self.FLAGS = FLAGS

        #self.netE = model.encoder(FLAGS.nei, FLAGS.nc, FLAGS.kW, FLAGS.sW, FLAGS.nef, FLAGS.nBottleneck)
        #self.netG = model.decoder(FLAGS.nBottleneck, FLAGS.noiseGen, FLAGS.nz, FLAGS.ngf, FLAGS.nc, FLAGS.ngo)
        #self.netD = model.discriminator(FLAGS.conditionAdv, FLAGS.nei, FLAGS.nc, FLAGS.kW, FLAGS.sW, FLAGS.ndf, FLAGS.ngo)
        self.netE = model.encoder_Michele(FLAGS.nei, FLAGS.nc, FLAGS.kW, FLAGS.sW)
        self.netG = model.decoder_Michele(FLAGS.nc, FLAGS.ngo)
        self.netD = model.discriminator_Michele(FLAGS.ngo, FLAGS.nc)

        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        self.vel_grad_loss_tracker = keras.metrics.Mean(name="vel_grad_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
        self.adversarial_loss_tracker = keras.metrics.Mean(name="adversarial_loss")
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")

        self.epoch_step = tf.Variable(0, trainable=False)

        self.global_train_batch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.train_summary_writer = tf.summary.create_file_writer(FLAGS.summary_dir + '/train')

        self.global_validation_batch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.validation_summary_writer = tf.summary.create_file_writer(FLAGS.summary_dir + '/validation')

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
            if self.FLAGS.pre_trained_generator:
                print("Restoring generator weights from {}".format(self.FLAGS.checkpoint))
                ckpt = tf.train.Checkpoint(netE=self.netE, netG=self.netG)
                status = ckpt.read(self.FLAGS.checkpoint)
                status.assert_existing_objects_matched()
            elif self.FLAGS.pre_trained_model:
                print("Restoring model weights from {}".format(self.FLAGS.checkpoint))
                ckpt = tf.train.Checkpoint(netE=self.netE, netG=self.netG, netD=self.netD)
                status = ckpt.read(self.FLAGS.checkpoint)
                status.assert_existing_objects_matched()
            else:
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

    def compile(self, d_optimizer, g_optimizer):
        super(TRGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, data):
        # Unpack the data.
        context_train, gap_train = data

        gap_train_pred = self(context_train, training=True)  # Predict the fake images

        # Train the discriminator
        with tf.GradientTape() as tape:
            if self.FLAGS.conditionAdv:
                discrim_fake_output = self.netD([context_train, gap_train_pred])
                discrim_real_output = self.netD([context_train, gap_train])
            else:
                discrim_fake_output = self.netD(gap_train_pred)
                discrim_real_output = self.netD(gap_train)

            discrim_fake_loss = tf.math.log(1 - discrim_fake_output + self.FLAGS.EPS)
            discrim_real_loss = tf.math.log(discrim_real_output + self.FLAGS.EPS)

            discrim_loss = tf.math.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

        # Log the discriminator output
        self.global_train_batch.assign_add(1)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('discriminator output', tf.math.reduce_mean(discrim_fake_output), step=self.global_train_batch)

        # Compute gradients
        trainable_vars = self.netD.trainable_variables
        gradients = tape.gradient(discrim_loss, trainable_vars)

        # Update weights
        self.d_optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
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

            # Adversarial loss
            if self.FLAGS.conditionAdv:
                discrim_fake_output = self.netD([context_train, gap_train_pred])
            else:
                discrim_fake_output = self.netD(gap_train_pred)

            adversarial_loss = tf.math.reduce_mean(-tf.math.log(discrim_fake_output + self.FLAGS.EPS))

            # Generator loss => content loss + adversarial loss
            gen_loss = (1 - self.FLAGS.adversarial_ratio) * content_loss + (self.FLAGS.adversarial_ratio) * adversarial_loss

        # Compute gradients
        trainable_vars = self.netE.trainable_variables + self.netG.trainable_variables
        gradients = tape.gradient(gen_loss, trainable_vars)

        # Update weights
        self.g_optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.mse_loss_tracker.update_state(mse_loss)
        self.vel_grad_loss_tracker.update_state(vel_grad_loss)
        self.content_loss_tracker.update_state(content_loss)
        self.adversarial_loss_tracker.update_state(adversarial_loss)
        self.gen_loss_tracker.update_state(gen_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.mse_loss_tracker, self.vel_grad_loss_tracker, self.content_loss_tracker, self.adversarial_loss_tracker, self.gen_loss_tracker]

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

        # Adversarial loss
        if self.FLAGS.conditionAdv:
            discrim_fake_output = self.netD([context_train, gap_train_pred])
        else:
            discrim_fake_output = self.netD(gap_train_pred)

        # Log the discriminator output
        self.global_validation_batch.assign_add(1)
        with self.validation_summary_writer.as_default():
            tf.summary.scalar('discriminator output', tf.math.reduce_mean(discrim_fake_output), step=self.global_validation_batch)

        adversarial_loss = tf.math.reduce_mean(-tf.math.log(discrim_fake_output + self.FLAGS.EPS))
        self.adversarial_loss_tracker.update_state(adversarial_loss)

        # Generator loss => content loss + adversarial loss
        gen_loss = (1 - self.FLAGS.adversarial_ratio) * content_loss + (self.FLAGS.adversarial_ratio) * adversarial_loss
        self.gen_loss_tracker.update_state(gen_loss)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}