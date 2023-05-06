#   import TensorFlow
import tensorflow as tf

sess = tf.Session()

# Verify we can print a string
hello = tf.constant("Hello Pluralsight from TensorFlow")
print(sess.run(hello))

#   Perform some simple math
a = tf.constant(20)
b = tf.constant(22)
print('a + b = {0}'.format(sess.run(a + b)))
