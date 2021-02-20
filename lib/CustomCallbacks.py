import io

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

def inpainting(context, gap, mask_gap):
    img = context.copy()
    img[:,mask_gap[0][0]:mask_gap[0][1],mask_gap[1][0]:mask_gap[1][1],:] = gap

    return img

def image_grid(imgs):
    """Return a rx3 grid of sampled images as a matplotlib figure."""
    r = imgs.shape[1]
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(12, r*2))

    titles = [None] * (r*3)
    titles[0] = "Damaged"
    titles[1] = "Generated"
    titles[2] = "Original"

    for i in range(r):
        for c in range(3):
            j = i*3 + (c+1)
            plt.subplot(r, 3, j, title=titles[j-1])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.contourf(np.swapaxes(imgs[c,i],0,1), np.linspace(-0.8,0.8,101), cmap=plt.cm.viridis)

    return figure

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class SampleCallback(keras.callbacks.Callback):
    def __init__(self, data_dev, mask_gap, logdir):
        super(SampleCallback, self).__init__()
        self.context_dev, self.gap_dev = data_dev
        self.mask_gap = mask_gap
        shape = self.context_dev.shape
        self.imgs = np.empty((3,)+shape[0:3])
        # Sets up the log directory.
        self.logdir = logdir
        self.file_writer = None

    def on_train_begin(self, logs=None):
        # Creates a file writer for the log directory.
        self.file_writer = tf.summary.create_file_writer(self.logdir)

    def on_epoch_end(self, epoch, logs=None):
        self.imgs[0] = self.context_dev[:,:,:,0]
        gap_pred = self.model.predict(self.context_dev)
        self.imgs[1] = inpainting(self.context_dev, gap_pred, self.mask_gap)[:,:,:,0]
        self.imgs[2] = inpainting(self.context_dev, self.gap_dev, self.mask_gap)[:,:,:,0]
        # Prepare the plot
        figure = image_grid(self.imgs)
        # Using the file writer, convert the plot to image and log
        with self.file_writer.as_default():
            tf.summary.image("samples_dev", plot_to_image(figure), step=epoch)


class EpochCallback(keras.callbacks.Callback):
    def __init__(self):
        super(EpochCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.epoch_step.assign_add(1)


class LearningRateSchedulerCallback(keras.callbacks.Callback):
    def __init__(self, boundaries, decay_rate):
        super(LearningRateSchedulerCallback, self).__init__()
        self.boundaries = boundaries
        self.decay_rate = decay_rate

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.boundaries:
            self.model.d_optimizer.learning_rate.assign(
                tf.math.multiply(self.model.d_optimizer.learning_rate, self.decay_rate)
            )

            self.model.g_optimizer.learning_rate.assign(
                tf.math.multiply(self.model.g_optimizer.learning_rate, self.decay_rate)
            )