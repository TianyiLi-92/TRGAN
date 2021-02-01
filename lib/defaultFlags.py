import tensorflow as tf

def defaultFlags():
    Flags = tf.compat.v1.flags

    # The system parameter
    Flags.DEFINE_string('output_dir', './output', 'The output directory of the checkpoint')
    Flags.DEFINE_string('summary_dir', './summary', 'The dirctory to output the summary')
    Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
    Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
    Flags.DEFINE_boolean('pre_trained_generator', False, 'If set True, the weight will be loaded but the global_step will still '
                     'be 0. If set False, you are going to continue the training. That is, '
                     'the global_step will be initiallized from the checkpoint, too')
    Flags.DEFINE_boolean('pre_trained_model', False, 'If set True, the weight will be loaded but the global_step will still '
                     'be 0. If set False, you are going to continue the training. That is, '
                     'the global_step will be initiallized from the checkpoint, too')
    Flags.DEFINE_string('task', 'TRCENet', 'The task: TRGAN, TRCENet')

    # The data preparing operation
    Flags.DEFINE_integer('batch_size', 64, 'Batch size of the input batch')
    Flags.DEFINE_string('dataset_path', './dataset/uf_y41.h5', 'The path of the dataset')
    Flags.DEFINE_list('mask_context', [[32,96],[32,96]], 'The size and position of the context')
    Flags.DEFINE_list('mask_gap', [[16,48],[16,48]], 'The size and position (relative to the context) of the gap')
    Flags.DEFINE_list('mask_GenOut', [[8,56],[8,56]], 'The size and position (relative to the context) of Gen output')

    # Generator configuration
    Flags.DEFINE_integer('nei', 16, 'The edge length of Encoder input')
    Flags.DEFINE_integer('nc', 1, 'Channels in input')
    Flags.DEFINE_integer('ekW', 4, 'The kernel width of the first convolution of Encoder')
    Flags.DEFINE_integer('esW', 4, 'The step of the first convolution of Encoder in the width dimension')
    Flags.DEFINE_integer('nef', 64, 'Encoder filters in first conv layer')
    Flags.DEFINE_integer('nBottleneck', 100, 'The dim for bottleneck of encoder')
    Flags.DEFINE_integer('ngf', 64, 'Gen filters in first conv layer')
    Flags.DEFINE_integer('dkW', 17, 'The kernel width of the last convolution of Decoder')
    Flags.DEFINE_integer('dsW', 1, 'The step of the last convolution of Decoder in the width dimension')
    Flags.DEFINE_integer('ngo', 8, 'The edge length of Gen output')
    Flags.DEFINE_boolean('noiseGen', False, 'Whether add noises Z of nz dimensions to the latent vector. True => Yes')
    Flags.DEFINE_string('noisetype', 'normal', 'Type of the noise: uniform / normal')
    Flags.DEFINE_integer('nz', 100, 'The dim for Z')

    # Adversarial discriminator configuration
    Flags.DEFINE_boolean('conditionAdv', False, 'Conditional GAN or not')
    Flags.DEFINE_integer('ndf', 64, 'Discrim filters in first conv layer')

    # The training parameters
    Flags.DEFINE_float('learning_rate', 0.0002, 'The learning rate for the network')
    Flags.DEFINE_float('beta', 0.5, 'The beta1 parameter for the Adam optimizer')
    Flags.DEFINE_integer('max_epoch', 25, 'The max epoch for the training')
    Flags.DEFINE_float('EPS', 1.e-12, 'Threshold for loss computations inside the log')
    Flags.DEFINE_float('adversarial_ratio', 1.e-3, 'Weighting factor for the adversarial loss')
    Flags.DEFINE_float('lambda_vel_grad', 0.2, 'Weighting factor for the velocity gradient in content loss')
    Flags.DEFINE_integer('overlapPred', 0, 'The width of overlapping edges')
    Flags.DEFINE_integer('globalSeed', 1234, 'The global random seed')

    FLAGS = Flags.FLAGS

    return FLAGS
