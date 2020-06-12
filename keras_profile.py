import os
import tensorflow as tf
import keras.backend as K
from models.firenet_tf import firenet_tf

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

def model_profile():
    model = firenet_tf(input_shape=(64, 64, 3))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    flops = get_flops(model)
    print(flops)
    return flops

def model_size(model_path):
    # size
    file_size = os.stat(model_path).st_size
    file_size /= 1024 **2
    fs_string = '{:.2f}MB'.format(file_size)
    print(fs_string)
    return fs_string


if __name__ == '__main__':
    # probing calculation stability
    model_profile()
    model_profile()
