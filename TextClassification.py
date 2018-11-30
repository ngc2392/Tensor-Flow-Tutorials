# https://www.tensorflow.org/tutorials/keras/basic_text_classification


"""
Classifies movie reviews as 'positive' or 'negative' using the text of the review.  This is an example of 'binary' - or two class - classification, an important and widely applicable kind of machine learning problem.

We'll use the IMBD dataset that contains the text of 50,000 movie reviews from the Internet Movie Database.  These are split into 25,000 reviews for training and 25,000 reviews for testing.  Teh training and testing sets are 'balanced', meaning tehy contain an equal number of positive and negative reviews.

This uses tf.keras, a high-level API to build and train models in TensorFlow.  For a more advanced text classification tutorial using tf.keras, see the MLCC Text Classification Guide.

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])



print(tf.__version__)


"""

DOWNLOAD THE IMBD DATASET

The IMBD dataset comes packages with TensorFlow.  It has already been preprocessed such that the reviews (sequences of words) have been converted to sequences of integers, where each integer represents a specific word in a dictionary.


"""

# The following code downloads the IMBD dataset to your machine (or uses a cached copy if you've already downloaded it)

imdb = keras.datasets.imdb

# The argument 'num_words=10000' keeps the top 10,000 most frequently occurring words in the training data.  The rare words are discarded to keep the size of the data manageable

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

"""

Let's take a moment to understand the format of the data.  The dataset comes preprocessed: each example is an array of integers representing the words of the omvie review.  Each label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.

"""

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# The text of reviews have been converrted to integers, where each integer represents a specific word in a dictionary.  Here's what the first review looks like: 

print(train_data[0])

# Movie reviews may be different lengths.  The below code shows the number of words in the first and second reviews.  Since inputs to a neural network must be the same length, we'll need to resolve this later.

print(len(train_data[0]), len(train_data[1]))

"""
CONVERT THE INTEGERS BACK TO WORDS

It may be useful to know how to convert integers back to text.  Here, we'll create a helper function to query a dictionary object that contains the integer to string mapping

"""


# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Now we can use the 'decode_review' function to display the text for the first review:

print(decode_review(train_data[0]))

"""
PREPARE THE DATA

The reviews - the arrays of integers - must be converted to tensors before fed into the neural network.  This conversation can be done a couple of ways:
    - One-hot-encode the arrays to convert them into vectors of 0s and 1s. F      or example, the sequence [3, 5] would become a 10,000-dimensional vect      or that is all zeros except for indices 3 and 5, which are ones. Then,      make this the first layer in our network—a Dense layer—that can handle      floating point vector data. This approach is memory intensive, though,      requiring a num_words * num_reviews size matrix.

    - Alternatively, we can pad the arrays so they all have the same length,    then create an integer tensor of shape max_length * num_reviews. We can     use an embedding layer capable of handling this shape as the first layer    in our network.

In this tutorial, we will use the second approach

"""

#Since the movie reviews must be the same length, we will use the 'pad_sequences' function to standardize the lengths

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# Let's look at the length of the examples now:

print(len(train_data[0]), len(train_data[1]))

# And inspect the (now padded) first review

print(train_data[0])

"""

The neural network is created by stacking layers - this requires two main architectural decisions:
    - How many layers to use in the model?
    - How many 'hidden units' to use for each layer?

In this example, the input data consists of an array of word-indices.  The labels to predict are either 0 or 1.

"""

# Let's build a model for this problem

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())


