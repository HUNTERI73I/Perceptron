import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpt
from keras import optimizers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#fashion_mnist = keras.datasets.fashion_mnist

mnist = keras.datasets.mnist

### Define Dataset
#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

### Define classes
#class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]



### Scaling pixels from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0 

plt.figure(figsize = (10, 10))


### Building Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),          ### Format pictures from 2d to 1d 28*28=784
    keras.layers.Dense(256, activation = tf.nn.relu),
    #keras.layers.Dense(196, activation = tf.nn.relu),
    #keras.layers.Dense(32, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

### Configuring model for training
model.compile(keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True),  ### Also possible Adadelta and Adagrad
              loss='sparse_categorical_crossentropy',                                              ### Do not change
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs = 10)         ### Experiments with epochs may be possible to decrease losses

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)

predictions = model.predict(test_images)

predictions[0]

### Visualization of predictions
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