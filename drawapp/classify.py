# Importing libraries
import cv2
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model


def classify(image, category):
    # print(image.shape)
    print(type(image))
    im = resize(image, (28, 28))
    im = np.array(im)
    im = np.reshape(im, (28, 28))
    im = np.expand_dims(im, axis=0)
    im = np.reshape(im, (28, 28, 1))
    # print(im)
    model = load_model('drawapp\\'+category+'.h5')
    prediction = model.predict(np.array([im]))
    list1 = prediction.tolist()[0]
    print(list1)
    index = np.argmax(list1)
    return index
