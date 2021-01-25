import tensorflow as tf
from tensorflow import keras

def periodic_padding(inpt, pad):
    ### Only work for pad > 0 ###
    L = inpt[:,:pad[0][0],:,:]
    R = inpt[:,-pad[0][1]:,:,:]
    inpt_pad = tf.concat([R, inpt, L], axis=1)

    L = inpt_pad[:,:,:pad[1][0],:]
    R = inpt_pad[:,:,-pad[1][1]:,:]
    inpt_pad = tf.concat([R, inpt_pad, L], axis=2)

    return inpt_pad


class PeriodicPadding2D(keras.layers.Layer):
    def __init__(self, pad, **kwargs):
        super(PeriodicPadding2D, self).__init__(**kwargs)
        self.pad = pad

    def call(self, inputs):
        return periodic_padding(inputs, self.pad)


def edge_cutting(inpt, edge):
    ### Only work for edge > 0 ###
    input_cutted = inpt[:,edge[0][0]:-edge[0][1],edge[1][0]:-edge[1][1],:]

    return input_cutted


class EdgeCutting2D(keras.layers.Layer):
    def __init__(self, edge, **kwargs):
        super(EdgeCutting2D, self).__init__(**kwargs)
        self.edge = edge

    def call(self, inputs):
        return edge_cutting(inputs, self.edge)


# =============================================================================
# def tf_ddx_fft(context, gap, mask):
#     shape = context.shape
# 
#     paddings = tf.constant([
#         [0, 0],
#         [mask[0][0], shape[1]-mask[0][1]],
#         [0, 0],
#         [0, 0]
#     ])
#     gap_pad = tf.pad(gap, paddings, "CONSTANT")
# 
#     inpt = tf.math.add(context[:, :, mask[1][0]:mask[1][1], :], gap_pad)
#     inpt = tf.transpose(inpt, perm=[0, 3, 2, 1])
# 
#     fft_length_h = shape[1] // 2
#     k = tf.concat([tf.range(fft_length_h), tf.constant([0])], 0)
#     ik = tf.complex( tf.zeros(fft_length_h+1), tf.scalar_mul(0.5, tf.cast(k, tf.float32)) )
#     ik = tf.reshape(ik, [1, 1, 1, fft_length_h+1])
# 
#     output = tf.signal.rfft(inpt)
#     output = tf.signal.irfft(tf.math.multiply(ik, output))
#     output = tf.transpose(output, perm=[0, 3, 2, 1])
# 
#     return output[:, mask[0][0]:mask[0][1], :, :]
# 
# 
# def tf_ddy_fft(context, gap, mask):
#     shape = context.shape
# 
#     paddings = tf.constant([
#         [0, 0],
#         [0, 0],
#         [mask[1][0], shape[2]-mask[1][1]],
#         [0, 0]
#     ])
#     gap_pad = tf.pad(gap, paddings, "CONSTANT")
# 
#     inpt = tf.math.add(context[:, mask[0][0]:mask[0][1], :, :], gap_pad)
#     inpt = tf.transpose(inpt, perm=[0, 1, 3, 2])
# 
#     fft_length_h = shape[2] // 2
#     k = tf.concat([tf.range(fft_length_h), tf.constant([0])], 0)
#     ik = tf.complex(tf.zeros(fft_length_h+1), tf.cast(k, tf.float32))
#     ik = tf.reshape(ik, [1, 1, 1, fft_length_h+1])
# 
#     output = tf.signal.rfft(inpt)
#     output = tf.signal.irfft(tf.math.multiply(ik, output))
#     output = tf.transpose(output, perm=[0, 1, 3, 2])
# 
#     return output[:, :, mask[1][0]:mask[1][1], :]
# =============================================================================


# =============================================================================
# def tf_wtl2Matrix(gap, overlapPred):
#     shape = tf.shape(gap)
# 
#     wtl2Matrix = tf.ones([
#          shape[0],
#          shape[1] - 2*overlapPred,
#          shape[2] - 2*overlapPred,
#          shape[3]
#     ])
#     paddings = tf.constant([
#         [0, 0],
#         [overlapPred, overlapPred],
#         [overlapPred, overlapPred],
#         [0, 0]
#     ])
#     wtl2Matrix = tf.pad(wtl2Matrix, paddings, "CONSTANT", constant_values=10)
# 
#     return wtl2Matrix
# =============================================================================



if __name__=="__main__":

    X = tf.constant([
        [[1,2,3],[4,5,6],[7,8,9]],
        [[11,12,13],[14,15,16],[17,18,19]],
    ])

    X = tf.expand_dims(X, axis=3)
    X_pad = PeriodicPadding2D( pad=((1,1),(1,1)) )(X)

    X = tf.squeeze(X, axis=3)
    X_pad = tf.squeeze(X_pad, axis=3)

    print('Before padding:')
    print(X.numpy())
    print('After padding:')
    print(X_pad.numpy())

    X_pad = tf.expand_dims(X_pad, axis=3)
    X_pad_cut = EdgeCutting2D( edge=((1,1),(1,1)) )(X_pad)

    X_pad = tf.squeeze(X_pad, axis=3)
    X_pad_cut = tf.squeeze(X_pad_cut, axis=3)

    print('After cutting:')
    print(X_pad_cut.numpy())
    print('Before cutting:')
    print(X_pad.numpy())