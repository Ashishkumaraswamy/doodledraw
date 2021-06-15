from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import RequestContext
from PIL import Image
import re
import base64
import io
import matplotlib.pyplot as plt
import numpy as np

# Create your views here.


def index(request):
    return render(request, "home.html")


def game(request):
    return render(request, "mainpage.html")


def get_canvas(request):
    if request.method == "POST":
        captured_image = request.POST['canvas_data']
        imgstr = re.search('base64,(.*)', captured_image).group(1)
        imgstr = base64.b64decode(imgstr)
        # print(imgstr)
        tempimg = io.BytesIO(imgstr)
        im = Image.open(tempimg)
        # print(np.array(im))
        im.show()
        return HttpResponse('')
