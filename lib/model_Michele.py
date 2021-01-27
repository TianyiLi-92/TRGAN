from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dropout, Reshape, Conv2DTranspose, Dense
from tensorflow.keras.models import Model

# Definition of the encoder (from Michele)
# Encode Input Context to noise (architecture similar to Discriminator)
def encoder_Michele(nei, nc, ekW, esW):
    input_context = Input(shape=(nei, nei, nc))
    # input is (nei) x (nei) x (nc)
    x = Conv2DTranspose(filters=3, kernel_size=ekW, strides=esW)(input_context)
    x = LeakyReLU(alpha=0.2)(x)
    # state size: 64 x 64 x 3
    x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 32 x 32 x 64
    x = Conv2D(64, (4, 4), (2, 2), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 16 x 16 x 64
    x = Conv2D(128, (4, 4), (2, 2), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 8 x 8 x 128
    x = Conv2D(256, (4, 4), (2, 2), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 4 x 4 x 256
    x = Flatten()(x)
    latent_vector = Dropout(rate=0.2)(x)
    # state size: 4096
    return Model(input_context, latent_vector)

# Definition of the decoder (from Michele)
# Decode noise to generate image
def decoder_Michele(nc, dkW, dsW, ngo):
    latent_vector = Input( (4096,) )
    # input is Z: 4096
    x = Reshape(target_shape=(4, 4, 256))(latent_vector)
    # state size: 4 x 4 x 256
    x = Conv2DTranspose(256, (4, 4), (2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 8 x 8 x 256
    x = Conv2DTranspose(128, (4, 4), (2, 2), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 16 x 16 x 128
    x = Conv2DTranspose(64, (4, 4), (2, 2), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 32 x 32 x 64
    x = Conv2D(3, (4, 4), (1, 1), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 32 x 32 x 3
    x = Conv2DTranspose(nc, (4, 4), (2, 2), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 64 x 64 x (nc)
    outputs = Conv2D(nc, dkW, dsW)(x)
    # state size: (ngo) x (ngo) x (nc)
    return Model(latent_vector, outputs)

# Adversarial discriminator net (from Michele)
def discriminator_Michele(ngo, dkW, dsW, nc):
    inputs = Input( (ngo, ngo, nc) )
    # input pred: (ngo) x (ngo) x (nc), going into a convolution
    x = Conv2DTranspose(nc, dkW, dsW)(inputs)
    # state size: 64 x 64 x (nc)
    x = Conv2D(3, (4, 4), (2, 2), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 32 x 32 x 3
    x = Conv2D(128, (4, 4), (2, 2), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 16 x 16 x 128
    x = Conv2D(256, (4, 4), (2, 2), 'same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 8 x 8 x 256
    x = Conv2D(512, (4, 4), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # state size: 8 x 8 x 512
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    # state size: 32768
    outputs = Dense(units=1, activation='sigmoid')(x)
    # state size: 1
    return Model(inputs, outputs)



if __name__=="__main__":
    import tensorflow as tf

    from defaultFlags import defaultFlags

    FLAGS = defaultFlags()

    netE = encoder_Michele(FLAGS.nei, FLAGS.nc, FLAGS.ekW, FLAGS.esW)
    netG = decoder_Michele(FLAGS.nc, FLAGS.dkW, FLAGS.dsW, FLAGS.ngo)
    netD = discriminator_Michele(FLAGS.ngo, FLAGS.dkW, FLAGS.dsW, FLAGS.nc)

    tf.keras.utils.plot_model(netE, to_file='netE_Michele.png', show_shapes=True)
    tf.keras.utils.plot_model(netG, to_file='netG_Michele.png', show_shapes=True)
    tf.keras.utils.plot_model(netD, to_file='netD_Michele.png', show_shapes=True)