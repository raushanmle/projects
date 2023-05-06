import tensorflow  as tf
sess = tf.Session()

with tf.name_scope("Equation_1"):
    a_e1 = tf.constant(10, name="a")
    b_e1 = tf.constant(15, name="b")
    c_e1 = tf.constant(5, name="c")

    x = tf.placeholder('int32', name="x")

    # z = ax**2 + bx + c
    z_e1 = tf.reduce_sum([tf.multiply(a_e1, tf.square(x)), tf.multiply(b_e1, x), c_e1], name="z")

with tf.name_scope("Equation_2"):
    a_e2 = tf.constant(10, name="a")
    b_e2 = tf.constant(15, name="b")
    c_e2 = tf.constant(5, name="c")

    x_e2 = tf.placeholder('int32', name="x")
    y_e2 = tf.placeholder('int32', name="y")

    # z = ax + by + c
    z_e2 = tf.reduce_sum([tf.multiply(a_e2, x_e2), tf.multiply(b_e2, y_e2), c_e2], name="z")

final_sum = tf.add(z_e1, z_e2)

eq1 = sess.run(z_e1, feed_dict={x:10})
print("Result of Equation_1: ", eq1)

eq2 = sess.run(z_e2, feed_dict={x_e2:10, y_e2:20})
print("Result of Equation_2: ", eq2)

fs = sess.run(final_sum, feed_dict={x:10, x_e2:10, y_e2:20})
print("Final sum is ", fs)

writer = tf.summary.FileWriter("./logs", sess.graph)
writer.close()
sess.close()