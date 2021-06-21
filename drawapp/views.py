from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import RequestContext
import re
import base64
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from drawapp.classify import classify
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


# Create your views here.
IMG_SIZE = 28
imgquescat = []
imgques = []
cnt = -1
majorcat = ['vehicle', 'utils','sports','shapes','art','human']
utils =  ['backpack','screwdriver','bucket','candle','cup','belt','hat','scissors','sword','tent']
sports = ['basketball', 'bat','drums','baseball bat','pool']
shapes = ['hexagon','octagon','circle','line','triangle']
art = ['pond', 'pool', 'popsicle','toilet','boomerang']
human =  ['hand','foot','eye','face','toe','tooth']
vehicle = ['airplane', 'aircraft carrier', 'bus', 'van',
           'truck', 'train', 'tractor', 'parachute', 'bicycle']

def normalize(data):
    "Takes a list or a list of lists and returns its normalized form"
    return np.interp(data, [0, 255], [-1, 1])

def random():
    global majorcat
    global imgques
    global cnt
    global imgquescat

    ques = np.random.choice(majorcat)
    if ques in imgquescat:
        random()
    else:
        imgquescat.append(ques)
        if ques == 'vehicles':
            imgques.append(np.random.choice(vehicle).upper())
        elif ques == 'sports':
            imgques.append(np.random.choice(sports).upper())
        elif ques == 'utils':
            imgques.append(np.random.choice(utils).upper())
        cnt += 1


def index(request):
    return render(request, "home.html")


def game(request):
    global cnt
    global imgques
    random()
    if cnt < 6:
        print(cnt)
        return render(request, "mainpage.html", {
            'imgques': imgques[cnt],
        })
    else:
        return render(request, "home.html")


def get_canvas(request):
    global imgques
    global cnt

    if request.method == "POST":
        captured_image = request.POST['canvas_data']
        imgstr = re.search('base64,(.*)', captured_image).group(1)
        imgstr = base64.b64decode(imgstr)
        with open('temp.png', 'wb') as output:
            output.write(imgstr)
        im = cv2.imread('temp.png' ,cv2.IMREAD_GRAYSCALE)
        
        # print(imgstr)
        # tempimg = io.BytesIO(imgstr)
        # im = Image.open(tempimg)
        # im = ImageOps.grayscale(im)
        # im.show()
        #im = np.array(imgstr)

        # cat = classify(im, imgquescat[cnt])
        x = cv2.resize(im, (28, 28))
        x = np.expand_dims(x, axis=0)
        x = np.reshape(x, (28, 28, 1))
        # invert the colors
        x = np.invert(x)
        # brighten the image by 60%
        for i in range(len(x)):
            for j in range(len(x)):
                if x[i][j] > 50:
                    x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)
        x = normalize(x)

        model = load_model('drawapp\\'+imgquescat[cnt]+'.h5')
        val = model.predict(np.array([x]))
        if imgquescat[cnt] == 'vehicles':
            global vehicle
            pred = vehicle[np.argmax(val)]
            print(pred)
            return HttpResponse(pred)
        elif imgquescat[cnt] == 'sports':
            global sports
            pred = sports[np.argmax(val)]
            print(pred)
            return HttpResponse(pred)
        elif imgquescat[cnt] == 'utils':
            global utils
            pred = utils[np.argmax(val)]
            print(pred)
            return HttpResponse(pred)
