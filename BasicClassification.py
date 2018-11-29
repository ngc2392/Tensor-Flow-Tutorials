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

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



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

# Let's explore the format of the dataset before training the model.  The following shows there are 60,000 images in the training set, with each image represnted as 28 x 28 pixels

#print(train_images.shape) #https://www.tensorflow.org/api_docs/python/tf/shape

#Output: (60000, 28, 28) The output tells us that there are 60,000 images in the training set, with each image represented as 28 x 28 pixels

#Likewise, there are 60,000 labels in the training set
#print(len(train_labels))

#Each label is an integer between 0 and 9
#print(train_labels)

# There are 10,000 images in the test set.  Again, each image is represented as 28 x 28 pixels
#print(test_images.shape)

# And the test set contains 10,000 images
#print(len(test_labels))

# The data must be preprocessed before training the network.  If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255

"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show() 
"""

# We scale these values to a range of 0 and 1 before feeding to the neural network model.  For this, cast the datatype of the image components from an integer to a float, and divide by 255.  It's important that the training set and the testing set are preprocessed in the same way

print("The type of train_images is " + str(type(train_images)))

train_images = train_images / 255.0
test_images = test_images / 255.0


# Display the first 25 images from the training set and display teh class name below each image.  Verify that the data is in the correct format and we're ready to build and train the network

"""
plt.figure(figsize=(10,10)) 
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""
# Building the neural network requires configuring the layers of the model, then compiling the model

#SETUP THE LAYERS

#The basic building bock of a neural network is the 'layer'.  Layers extract representations from the data fed into them.  And, hopefully, these representations are more meaningful for the problem at hand.

# Most of deep learning consists of chaining together simple layers.  MOst layers, like tf.keras.layers.Dense, have parameters that are learned during training. 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""

The first layer of this network, tf.keras.layers.Flatten, transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.  Think of this layer as unstacking rows of pixels in the image and lining them up.  This layer has no parameters to learn; it only refomats the data

After the pixels are flattened, the netwrok consists of a sequence of two tf.keras.layers.Dense layers.  These are densely-connected, or fully-connected,neural layers.  The first 'Dense' layer has 128 nodes (or neurons).  The second (and last) layer is a 10-node 'softmax' layer-this returns an array of 10 probability scores that sum to 1.  Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.  

COMPILE THE MODEL 

Before the model is ready for training, it needs a few more settings.  These are added during the model's 'compile' step:
    - Loss function - This measures how accurate the model is during training.  We want to minimize this function to "steer" the model in the right direction.
    - Optimizer - This is how the model is updated based on the data it sees and its loss function
    - Metrics - Used to monitor the training and testing steps.  The following example uses 'accuracy', the fraction of the images that are correctly classified.

"""


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""

TRAIN THE MODEL

Training the neural network model requires the following steps:
    1. Feed the training data to the model-in this example, the train_images and train_labels arrays
    2. The model learns to associate images and labels
    3. We ask the model to make predictions about a test set - in this example, the test_images array.  We verify that the predictions match the labels from the test_labels array

"""

# To start training, call the model.fit method - the method is "fit" to the training data:

model.fit(train_images, train_labels, epochs=5)

# As the model trains, the loss and accuracy metrics are displayed.  This model reaches an accuracy of about 0.88 (or 88%) on the training data.

#EVALUATE ACCURACY

# Next, compare how the model performs on the test dataset

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset.  This gap between training accuracy and test accuracy is an example of 'overfitting'.  Overfitting is when a machine llearning model performs worse on new data than on their training data.

#MAKE PREDICTIONS

# With the model trained, we can use it to make predictoins about some images

predictions = model.predict(test_images)

# Here, the model has predicted the lael for each image in teh testing set.  Let's take a look at the first prediction:

print(predictions[0])

# A prediction is an array of 10 numbers.  These describe the "confidence" of the model that the image corresponds to each of the 10 different articles of clothing.  We can see which label has the highest confidence value:

print(np.argmax(predictions[0]))

#So the model is most confident that this image is an ankle boot, or class_names[9].  And we can check the test label to see if this is correct

print(test_labels[0])

# We can graph this to look at the full set of 10 channels

#Let's look at the 0th image, predictions, and prediction array
"""
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

"""
# Let's look at the 12th image, 
"""
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()
"""
# Let's plot several images with their predictions.  Correct predictions labels are blue and incorrect prediction labels are red.  The number gives the percent (out of 100) for the predicted label.  Note that it can be wrong even when very confident.


# Plot the first X test images, their predicted label, and the true label
# Color correct predcitions in blue, incorrect predictions in red


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Finally, use the trained model to make a prediction about a single image

#Grab an image from the test dataset

img = test_images[0]

print(img.shape)

# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once.  So even though we're using a single image, we need to add it to a list:
img = (np.expand_dims(img, 0))

print(img.shape)

# Now predict the image:
predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# model.predict returns a list of lists, one for each image in the batch of data.  Grab the predictions for our (only) image in the batch
print(np.argmax(predictions_single[0]))

# And, as before, the model predicts a label of 9