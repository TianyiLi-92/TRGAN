import os
from tensorflow.keras.optimizers import Adam

import numpy as np
import time

from lib.defaultFlags import defaultFlags
from lib.generateDataset import generateDataset
from lib.TRCENet import TRCENet
from lib.TRGAN import TRGAN

FLAGS = defaultFlags()

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

if FLAGS.task == 'TRCENet':
    net = TRCENet(FLAGS)
    net.compile( optimizer=Adam(learning_rate=FLAGS.learning_rate, beta_1=FLAGS.beta) )
elif FLAGS.task == 'TRGAN':
    net = TRGAN(FLAGS)
    d_optimizer = Adam(FLAGS.learning_rate/2., FLAGS.beta)
    g_optimizer = Adam(FLAGS.learning_rate, FLAGS.beta)
    net.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer)
else:
    raise ValueError('Need to specify FLAGS.task to be TRCENet or TRGAN')

context_train, gap_train, context_dev, gap_dev, context_test, gap_test = generateDataset(FLAGS.dataset_path, FLAGS.mask_context, FLAGS.mask_gap)

hist_train, bin_edges_train = np.histogram(gap_train, bins=100, density=True)
hist_dev, bin_edges_dev = np.histogram(gap_dev, bins=100, density=True)

KLD_train, KLD_dev = [], [] # Kullback-Leibler divergence

for epoch in range(FLAGS.max_epoch):
    print("\nStart of epoch %d" % (epoch+1,))
    start_time = time.time()

    net.load_weights( os.path.join(FLAGS.output_dir, 'checkpoint', 'ckpt.{:02d}'.format(epoch+1)) )

    gap_train_pred = net.predict(context_train, verbose=2)
    hist_train_pred, _ = np.histogram(gap_train_pred, bins=100, density=True)

    KLD = np.sum( hist_train * np.log( hist_train / (hist_train_pred + FLAGS.EPS) ) * (bin_edges_train[1] - bin_edges_train[0]) )
    KLD_train.append(KLD)

    gap_dev_pred = net.predict(context_dev, verbose=2)
    hist_dev_pred, _ = np.histogram(gap_dev_pred, bins=100, density=True)

    KLD = np.sum( hist_dev * np.log( hist_dev / (hist_dev_pred + FLAGS.EPS) ) * (bin_edges_dev[1] - bin_edges_dev[0]) )
    KLD_dev.append(KLD)

    print("Epoch %d/%d - %ds" % (epoch + 1, FLAGS.max_epoch, time.time() - start_time))

import pandas as pd
df_KLD = pd.DataFrame({'KLD_train': KLD_train, 'KLD_dev': KLD_dev})
df_KLD.to_csv(os.path.join(FLAGS.output_dir, 'KLD.csv'), index=False)

print("--------------- ******** -----------------")
