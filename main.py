import os
import h5py
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from lib.defaultFlags import defaultFlags
from lib.generateDataset import generateDataset
from lib.TRCENet import TRCENet
from lib.TRGAN import TRGAN
from lib.CustomCallbacks import SampleCallback, EpochCallback

FLAGS = defaultFlags()

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# Check the summary_dir is given
if FLAGS.summary_dir is None:
    raise ValueError('The summary directory is needed')

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

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

context_train, gap_train, context_dev, gap_dev, context_test, gap_test = generateDataset(FLAGS.dataset_path, FLAGS.mask_context, FLAGS.mask_gap, FLAGS.mask_GenOut)

print("--------------- {} mode -----------------".format(FLAGS.mode))
tf.random.set_seed(FLAGS.globalSeed)
net.initialize()
print("Finished initializing :D")

if FLAGS.mode == 'train':
    #early_stopping_callback = EarlyStopping(monitor='val_content_loss', patience=50, verbose=2, mode='min')
    model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(FLAGS.output_dir, 'checkpoint', 'ckpt.{epoch:02d}'))
    tensorboard_callback = TensorBoard(log_dir=FLAGS.summary_dir, write_graph=False, profile_batch=0)
    sample_callback = SampleCallback(data_dev=(context_dev[0:3], gap_dev[0:3]), mask_gap=FLAGS.mask_gap, mask_GenOut=FLAGS.mask_GenOut, logdir=os.path.join(FLAGS.summary_dir, 'image'))
    epoch_callback = EpochCallback()

    #callbacks = [early_stopping_callback, model_checkpoint_callback, tensorboard_callback, sample_callback, epoch_callback]
    callbacks = [model_checkpoint_callback, tensorboard_callback, sample_callback, epoch_callback]

    train_history = net.fit(context_train, gap_train, batch_size=FLAGS.batch_size, epochs=FLAGS.max_epoch, verbose=2, callbacks=callbacks, validation_data=(context_dev, gap_dev), shuffle=True, initial_epoch=net.epoch_step.numpy())

    import pandas as pd
    df_train_history = pd.DataFrame(train_history.history)
    df_train_history.to_csv(os.path.join(FLAGS.output_dir, 'train_history.csv'), index=False)

elif FLAGS.mode == 'test':
    net.evaluate(context_dev, gap_dev, batch_size=FLAGS.batch_size, verbose=2)
    gap_dev_pred = net.predict(context_dev, verbose=2)

    net.evaluate(context_test, gap_test, batch_size=FLAGS.batch_size, verbose=2)
    gap_test_pred = net.predict(context_test, verbose=2)

    # Discriminator output
    if FLAGS.conditionAdv:
        discrim_fake_output_dev = net.netD.predict([context_dev, gap_dev_pred])
    else:
        discrim_fake_output_dev = net.netD.predict(gap_dev_pred)

    if FLAGS.conditionAdv:
        discrim_fake_output_test = net.netD.predict([context_test, gap_test_pred])
    else:
        discrim_fake_output_test = net.netD.predict(gap_test_pred)

    # Check the output directory to save the prediction
    prediction_dir = os.path.join(FLAGS.output_dir, 'prediction')
    if not os.path.exists(prediction_dir):
        os.mkdir(prediction_dir)

    filename_out = os.path.join(prediction_dir, os.path.basename(FLAGS.checkpoint) +'.h5')
    print("Saving dev data and test data to {}".format(filename_out))
    with h5py.File(filename_out, 'w') as h5f:
        h5f.create_dataset('context_dev', data=context_dev)
        h5f.create_dataset('gap_dev', data=gap_dev)
        h5f.create_dataset('gap_dev_pred', data=gap_dev_pred)

        h5f.create_dataset('context_test', data=context_test)
        h5f.create_dataset('gap_test', data=gap_test)
        h5f.create_dataset('gap_test_pred', data=gap_test_pred)

        h5f.create_dataset('discrim_fake_output_dev', data=discrim_fake_output_dev)
        h5f.create_dataset('discrim_fake_output_test', data=discrim_fake_output_test)

print("--------------- ******** -----------------")
