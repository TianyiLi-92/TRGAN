from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Conv2DTranspose, ReLU, Concatenate, ZeroPadding2D, Activation, Reshape
from tensorflow.keras.models import Model

from lib.ops import PeriodicPadding2D, EdgeCutting2D
#from ops import PeriodicPadding2D, EdgeCutting2D

# Definition of the encoder
# Encode Input Context to noise (architecture similar to Discriminator)
def encoder(nei, nc, kW, sW, nef, nBottleneck):
    input_context = Input(shape=(nei, nei, nc))
    # input is nei x nei x (nc)
    x = Conv2DTranspose(filters=nc, kernel_size=2*kW, strides=2*sW)(input_context)
    x = LeakyReLU(alpha=0.2)(x)
    # state size: 128 x 128 x (nc)
    x = PeriodicPadding2D( pad=((1,1),(1,1)) )(x)
    x = Conv2D(filters=nef, kernel_size=(4, 4), strides=(2, 2))(x)
    x = LeakyReLU(alpha=0.2)(x)
    # state size: 64 x 64 x (nef)
    x = Conv2D(nef, (4, 4), (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # state size: 32 x 32 x (nef)
    x = Conv2D(nef * 2, (4, 4), (2, 2), 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # state size: 16 x 16 x (nef*2)
    x = Conv2D(nef * 4, (4, 4), (2, 2), 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # state size: 8 x 8 x (nef*4)
    x = Conv2D(nef * 8, (4, 4), (2, 2), 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # state size: 4 x 4 x (nef*8)
    latent_vector = Conv2D(nBottleneck, (4, 4))(x)
    # state size: 1 x 1 x (nBottleneck)
    return Model(input_context, latent_vector)

# Definition of the decoder
def decoder(nBottleneck, noiseGen, nz, ngf, nc, ngo):
    latent_vector = Input( (1, 1, nBottleneck) )
    # input is Z: 1 x 1 x (nBottleneck)

    if noiseGen:
        noise = Input( (1, 1, nz) )
        # input is Z: 1 x 1 x (nz), going into a convolution
        x = Conv2D(nz, (1, 1), (1, 1))(noise)
        # state size: 1 x 1 x (nz)
        x = Concatenate()([latent_vector, x])
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # state size: 1 x 1 x (nBottleneck+nz)

        nz_size = nBottleneck+nz
    else:
        x = BatchNormalization()(latent_vector)
        x = LeakyReLU(0.2)(x)

        nz_size = nBottleneck

    # Decode noise to generate image
    # input is Z: 1 x 1 x (nz_size), going into a convolution
    x = Conv2DTranspose(ngf * 8, (4, 4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # state size: 4 x 4 x (ngf*8)
    x = Conv2DTranspose(ngf * 4, (4, 4), (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # state size: 8 x 8 x (ngf*4)
    x = Conv2DTranspose(ngf * 2, (4, 4), (2, 2), 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # state size: 16 x 16 x (ngf*2)
    x = Conv2DTranspose(ngf, (4, 4), (2, 2), 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # state size: 32 x 32 x (ngf)
    x = Conv2DTranspose(nc, (4, 4), (2, 2), 'same')(x)
    # state size: 64 x 64 x (nc)
    outputs = Conv2D(nc, 64//ngo, 64//ngo, activation='tanh')(x)
    # state size: (ngo) x (ngo) x (nc)

    if noiseGen:
        return Model([latent_vector, noise], outputs)
    else:
        return Model(latent_vector, outputs)

# Adversarial discriminator net
def discriminator(conditionAdv, nei, nc, kW, sW, ndf, ngo):
    if conditionAdv:
        input_context = Input( (nei, nei, nc) )
        # input Context: (nei) x (nei) x (nc), going into a convolution
        x_context = Conv2DTranspose(nc, 2*kW, 2*sW)(input_context)
        # state size: 128 x 128 x (nc)
        x_context = ZeroPadding2D(padding=(2, 2))(x_context)
        x_context = Conv2D(ndf, (5, 5), (2, 2))(x_context)
        # state size: 64 x 64 x (ndf)

        input_pred = Input( (ngo, ngo, nc) )
        # input pred: (ngo) x (ngo) x (nc), going into a convolution
        x_pred = Conv2DTranspose(nc, 64//ngo, 64//ngo)(input_pred)
        # state size: 64 x 64 x (nc)
        x_pred = ZeroPadding2D((2+32, 2+32))(x_pred)  # 32: to keep scaling of features same as context
        x_pred = Conv2D(ndf, (5, 5), (2, 2))(x_pred)
        # state size: 64 x 64 x (ndf)

        x = Concatenate(axis=3)([x_context, x_pred])
        x = LeakyReLU(0.2)(x)
        # state size: 64 x 64 x (ndf * 2)

        x = Conv2D(ndf, (4, 4), (2, 2), 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        # state size: 32 x 32 x (ndf)
    else:
        inputs = Input( (ngo, ngo, nc) )
        # input pred: (ngo) x (ngo) x (nc), going into a convolution
        x = Conv2DTranspose(nc, 64//ngo, 64//ngo)(inputs)
        # state size: 64 x 64 x (nc)
        x = Conv2D(ndf, (4, 4), (2, 2), 'same')(x)
        x = LeakyReLU(0.2)(x)
        # state size: 32 x 32 x (ndf)

    x = Conv2D(ndf * 2, (4, 4), (2, 2), 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # state size: 16 x 16 x (ndf*2)
    x = Conv2D(ndf * 4, (4, 4), (2, 2), 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # state size: 8 x 8 x (ndf*4)
    x = Conv2D(ndf * 8, (4, 4), (2, 2), 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # state size: 4 x 4 x (ndf*8)
    x = Conv2D(1, (4, 4))(x)
    x = Activation('sigmoid')(x)
    # state size: 1 x 1 x 1
    outputs = Reshape((1,))(x)
    # state size: 1

    if conditionAdv:
        return Model([input_context, input_pred], outputs)
    else:
        return Model(inputs, outputs)



if __name__=="__main__":
    import tensorflow as tf

    from defaultFlags import defaultFlags

    FLAGS = defaultFlags()

    netE = encoder(FLAGS.nei, FLAGS.nc, FLAGS.kW, FLAGS.sW, FLAGS.nef, FLAGS.nBottleneck)
    netG = decoder(FLAGS.nBottleneck, FLAGS.noiseGen, FLAGS.nz, FLAGS.ngf, FLAGS.nc, FLAGS.ngo)
    netD = discriminator(FLAGS.conditionAdv, FLAGS.nei, FLAGS.nc, FLAGS.kW, FLAGS.sW, FLAGS.ndf, FLAGS.ngo)

    tf.keras.utils.plot_model(netE, to_file='netE.png', show_shapes=True)
    tf.keras.utils.plot_model(netG, to_file='netG.png', show_shapes=True)
    tf.keras.utils.plot_model(netD, to_file='netD.png', show_shapes=True)
