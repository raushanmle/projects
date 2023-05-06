
## Working with Images on TensorFlow

**1. Load Image.py**

&nbsp;&nbsp;&nbsp; This python script helps in loading images from a directory and transposing them. It uses openCV library to read images into a numpy array.

**2. Resizing Images**

&nbsp;&nbsp;&nbsp; This python script helps in loading images from a directory and resizing them. It uses openCV library to read images into a numpy array.

**3. Showing Images on TensorBoard**

&nbsp;&nbsp;&nbsp; To write image into Tf summary, a 4-D tensor is created, which can be thought of as a list of images. This 4-D tensor requires all the images to be of the same shape. Therefore, this code loads images from a directory, resizes them into 225x225 pixel images and stores them into a 4D array.
The Tf summary stores this images which can be seen on the TensorBoard.

![](https://github.com/shubham99bisht/Learning-TensorFlow/blob/master/3.Working with Images/Screenshot.png)