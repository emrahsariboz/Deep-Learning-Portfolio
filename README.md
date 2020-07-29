# Deep-Learning-Portfolio
There are two main reasons I created this repository.
1) Show off my deep learning projects.
2) Strengthen my understanding of deep learning concepts while trying to explain others. 


# Convolutional Neural Network

Convolution. It is a bit weird word, isn't it?  Originally, it refers to a mathematical combination of two functions to produce third functions. We will come to what it refers to in terms of deep learning in a bit. Before that, let's talk about why CNN?

CNN is a deep learning algorithm that differs from traditional multi-layer perceptrons(MLP) in a couple of smart ways. In MLP (also known as vanilla neural network), we have an input vector that is fully connected along with associated weights to the hidden layer, which is again connected to another hidden layer and so on until we get to the final/output layer. MLP uses a perceptron for each pixel in an image and assigns a weight to each perceptron. With real word image classification problems, the number of pixels increases, which ultimately makes the MLP  undesirable approach. The chance of overfitting arises as each layer is fully connected to the previous and next layer. 

# How does it work?

As we stated above, CNN expects an image rather than a flattened image vector like MLP does. Every image pixel has value ranges between 0 and 255. 0 stands for black, where 255 is white. If the image you feed to the network is grayscale, you will only have one channel. The number of channels will increase to three if you decide to classify a picture that is not a grayscale (real-world intricacy). Channel represents RGB (Red, Green, Blue), and color can be found by combining three layers. 

The very first step once the input image is resized to common shape is to pass it through the convolution layer. The purpose of the convolution layer is to detect features of your images using a feature detector called a Kernel, a.k.a. Filter. A Filter is a grip of numbers that has a shape that slides over the input image. At every slide, element-wise multiplication is performed and summed to a single value. Imagine a small filter sliding from left to right across the image from top to bottom. 


Using filters, we can filter out irrelevant information or distinguish the object boundaries or other relevant features. 


# Autograd






