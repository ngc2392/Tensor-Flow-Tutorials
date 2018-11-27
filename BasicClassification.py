#https://www.tensorflow.org/tutorials/keras/basic_classification

"""

This guide trains a neural network model to classify images of clothing, like sneakers and shirts.  This guide uses tf.keras, a  high-level API to build and train models in TensorFlow

"""
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt # https://matplotlib.org/users/installing.html#linux-using-your-package-manager

#print(dir(tf)) 

# print(tf.__version__)

#This guide uses the Fashion MNIST dataset https://github.com/zalandoresearch/fashion-mnist
# We will use 60,000 images to train the network and 10,000 images to evaluate how accurately the network learned to classify images.  You can access the Fashion MNIST directly from TensorFlow, just import and load the data.


fashion_mnist = keras.datasets.fashion_mnist 

"""
Loading the dataset returns four NumPy arrays: (returns 2 tuples)

	train_images and train_labels arrays are the 'training set' - the data the model uses to learn

	The model is tested against the 'test set', the test_images and test_labels array

	The images are 28x28 NumPy arrays, with pixel values ranging from between 0 and 255. The 'labels' are an array of integers, ranging from 0 to 9 

	Each image is mapped to a single label	

"""

# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data or https://keras.io/datasets

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Store the class names here for later, since they are not included with the dataset.  We will use them later to plot the images

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Let's explore the format of the dataset before training the model.  The follo
