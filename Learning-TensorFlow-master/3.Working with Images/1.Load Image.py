import tensorflow as tf
import cv2

image = cv2.imread("./Images/flower2.jpg")
print(image.shape)

x = tf.Variable(image, name = "x")
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    transpose = tf.image.transpose_image(x)

    res = sess.run(transpose)

    print(res.shape)
    cv2.imshow("Image",res)
    cv2.waitKey(0)

cv2.destroyAllWindows()