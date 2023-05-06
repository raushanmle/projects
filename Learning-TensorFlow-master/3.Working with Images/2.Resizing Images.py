import tensorflow as tf
import cv2
import numpy as np

image_names = ["flower1.jpg", "flower2.jpg", "flower3.jpg"]


all_images = []

with tf.Session() as sess:

    for i in range(len(image_names)):
        image = cv2.imread("./Images/"+image_names[i])
        x = tf.Variable(image, name = "x")
        init = tf.global_variables_initializer()
        sess.run(init)

        resize = tf.image.resize_images(x, [225, 225])
        res = sess.run(resize)

        print(res.shape, type(res))
        all_images.append(res)

    print(type(all_images), len(all_images))
    all_images = np.asarray(all_images)

    # A 4D tensor which stores 3 images of the shape 225x225.
    image_tensor = tf.convert_to_tensor(all_images)
    print(type(image_tensor), image_tensor.shape)




