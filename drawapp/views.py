from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import RequestContext
from PIL import Image
import re
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from drawapp.classify import classify
from PIL import Image, ImageOps


# Create your views here.

imgquescat = []
imgques = []
cnt = -1
majorcat = ['vehicles', 'utils']
utils = ['backpack', 'screwdriver', 'bucket', 'candle', 'cup']
sports = ['basketball', 'bat', 'drums', 'baseball bat', 'pool']
vehicle = ['airplane', 'aircraft carrier', 'bus', 'van',
           'truck', 'train', 'tractor', 'parachute', 'bicycle']


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
        # print(imgstr)
        tempimg = io.BytesIO(imgstr)
        im = Image.open(tempimg)
        im = ImageOps.grayscale(im)
        im.show()
        im = np.array(im)

        cat = classify(im, imgquescat[cnt])
        if imgquescat[cnt] == 'vehicles':
            global vehicle
            print(vehicle[cat])
            return HttpResponse(vehicle[cat])
        elif imgquescat[cnt] == 'sports':
            global sports
            print(sports[cat])
            return HttpResponse(sports[cat])
        elif imgquescat[cnt] == 'utils':
            global utils
            print(utils[cat])
            return HttpResponse(utils[cat])
