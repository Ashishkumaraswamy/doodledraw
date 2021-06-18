# Importing libraries
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import warnings
import pickle
import weakref
from sklearn. model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
from skimage.transform import resize
from skimage.io import imread
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
DATA_DIR = 'E:/sketchimg/png'
# X = []
# y = []
# cnt = 1
# CATEGORY_DIR_LIST_PATH = ['airplane', 'alarm clock', 'angel', 'ant', 'apple', 'arm',
#                           'armchair', 'ashtray', 'axe', 'backpack', 'banana', 'barn',
#                           'baseball bat', 'basket', 'bathtub', 'bear (animal)', 'bed', 'bee',
#                           'beer-mug', 'bell', 'bench', 'bicycle', 'binoculars', 'blimp',
#                           'book', 'bookshelf', 'boomerang', 'bottle opener', 'bowl', 'brain',
#                           'bread', 'bridge', 'bulldozer', 'bus', 'bush', 'butterfly',
#                           'cabinet', 'cactus', 'cake', 'calculator', 'camel', 'camera',
#                           'candle', 'cannon', 'canoe', 'car (sedan)', 'carrot', 'castle',
#                           'cat', 'cell phone', 'chair', 'chandelier', 'church', 'cigarette',
#                           'cloud', 'comb', 'computer monitor', 'computer-mouse', 'couch',
#                           'cow', 'crab', 'crane (machine)', 'crocodile', 'crown', 'cup',
#                           'diamond', 'dog', 'dolphin', 'donut', 'door', 'door handle',
#                           'dragon', 'duck', 'ear', 'elephant', 'envelope', 'eye',
#                           'eyeglasses', 'face', 'fan', 'feather', 'fire hydrant', 'fish',
#                           'flashlight', 'floor lamp', 'flower with stem', 'flying bird',
#                           'flying saucer', 'foot', 'fork', 'frog', 'frying-pan', 'giraffe',
#                           'grapes', 'grenade', 'guitar', 'hamburger', 'hammer', 'hand',
#                           'harp', 'hat', 'head', 'head-phones', 'hedgehog', 'helicopter',
#                           'helmet', 'horse', 'hot air balloon', 'hot-dog', 'hourglass',
#                           'house', 'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo',
#                           'key', 'keyboard', 'knife', 'ladder', 'laptop', 'leaf',
#                           'lightbulb', 'lighter', 'lion', 'lobster', 'loudspeaker',
#                           'mailbox', 'megaphone', 'mermaid', 'microphone', 'microscope',
#                           'monkey', 'moon', 'mosquito', 'motorbike', 'mouse (animal)',
#                           'mouth', 'mug', 'mushroom', 'nose', 'octopus', 'owl', 'palm tree',
#                           'panda', 'paper clip', 'parachute', 'parking meter', 'parrot',
#                           'pear', 'pen', 'penguin', 'person sitting', 'person walking',
#                           'piano', 'pickup truck', 'pig', 'pigeon', 'pineapple',
#                           'pipe (for smoking)', 'pizza', 'potted plant', 'power outlet',
#                           'present', 'pretzel', 'pumpkin', 'purse', 'rabbit', 'race car',
#                           'radio', 'rainbow', 'revolver', 'rifle', 'rollerblades', 'rooster',
#                           'sailboat', 'santa claus', 'satellite', 'satellite dish',
#                           'saxophone', 'scissors', 'scorpion', 'screwdriver', 'sea turtle',
#                           'seagull', 'shark', 'sheep', 'ship', 'shoe', 'shovel',
#                           'skateboard', 'skull', 'skyscraper', 'snail', 'snake', 'snowboard',
#                           'snowman', 'socks', 'space shuttle', 'speed-boat', 'spider',
#                           'sponge bob', 'spoon', 'squirrel', 'standing bird', 'stapler',
#                           'strawberry', 'streetlight', 'submarine', 'suitcase', 'sun', 'suv',
#                           'swan', 'sword', 'syringe', 't-shirt', 'table', 'tablelamp',
#                           'teacup', 'teapot', 'teddy-bear', 'telephone', 'tennis-racket',
#                           'tent', 'tiger', 'tire', 'toilet', 'tomato', 'tooth', 'toothbrush',
#                           'tractor', 'traffic light', 'train', 'tree', 'trombone',
#                           'trousers', 'truck', 'trumpet', 'tv', 'umbrella', 'van', 'vase',
#                           'violin', 'walkie talkie', 'wheel', 'wheelbarrow', 'windmill',
#                           'wine-bottle', 'wineglass', 'wrist-watch', 'zebra']
# CATEGORY_DIR_LIST_PATH = CATEGORY_DIR_LIST_PATH[:5]
# for catpath in CATEGORY_DIR_LIST_PATH:
#     pathtoffile = os.path.join(DATA_DIR, catpath)
#     for img in os.listdir(pathtoffile):
#         finpath = os.path.join(pathtoffile, str(cnt)+'.png')
#         print(cnt)
#         try:
#             imgread = imread(finpath)
#         except FileNotFoundError as e:
#             pass
#         cnt += 1
#         imgread = resize(imgread, (32, 32, 3))
#         # plt.imshow(imgread)
#         X.append(imgread.flatten())
#         y.append(CATEGORY_DIR_LIST_PATH.index(catpath))
# X = np.array(X)
# y = np.array(y)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42)
# param_grid = [
#     {'C': [1, 10, 100, 1000], 'kernel':['linear']},
#     {'C': [1, 10, 100, 1000], 'gamma':[0.001, 0.001], 'kernel':['rbf']},
# ]
# svc = svm.SVC(probability=True)
# clf = GridSearchCV(svc, param_grid)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_pred
# accuracy_score(y_pred, y_test)
# pickle.dump(clf, open("modelsvm2.p", 'wb'))


def classify(image, category):
    image = resize(image, (32, 32, 3))
    # print(image)
    # list1 = []
    # list1.append(image)
    # inputimg = np.array(list1)
    # inputimg = inputimg/255
    inputimg = image.flatten()
    inputimg = inputimg.reshape(1, 3072)
    # model = load_model('drawapp\my_model.h5')
    model = pickle.load(open("modelsvm2.p", 'rb'))
    prediction = model.predict(inputimg)
    # maxval = np.max(prediction)
    # print(maxval)
    list1 = prediction.tolist()[0]
    index = np.argsort(list1)[-3:][::-1]
    # str1 = ''
    # str1 += category[index[0]]+' '+category[index[1]]+' '+category[index[2]]
    # print(str1)
    return category[int(prediction)]
