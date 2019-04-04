import tensorflow as tf
from tensorflow.python.client import device_lib


if tf.test.gpu_device_name():
    print 'Default GPU Device: %s' % tf.test.gpu_device_name()
else:
    print 'Please install GPU version of TF'

print device_lib.list_local_devices()
