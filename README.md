# Image_Colorization
An Autoencoder made with Keras that can colorize any black and white image.
I first converted all images present in the cifar10 image dataset into greyscale images. The autoencoder was then trained on a dataset where the inputs were the greyscale images and their outputs were the colorized original images.
Here's the colorized output of the first 100 greyscale test images of the cifar10 dataset-

![color_img](https://user-images.githubusercontent.com/36446402/62408392-17540a80-b5e6-11e9-82a5-3bf4eee92def.png)
