# MNIST dataset | Tensorflow

MNIST is a dataset of handwritten digits. It contains 60,000 training images and 10,000 testing images of size (28, 28).

![MNIST examples](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/320px-MnistExamples.png)

### Models

* `mnist_simple.py`: Simple model using Dense layers

* `mnist_conv.py` : CNN model using Conv2D and Dense layers

### Accuracy

`mnist_simple.py:`

train: 98.2%  
test: 97.2%

`mnist_conv.py:`

train: 99.4%  
test: 98.8%
