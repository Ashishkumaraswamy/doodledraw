from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
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
    if request.method == "POST":
        print("Hi")
        if request.POST.get('captured_image'):
            captured_image = request.POST.get('captured_image')
            imgstr = re.search('base64,(.*)', captured_image).group(1)
            imgstr = base64.b64decode(imgstr)
            # print(imgstr)
            tempimg = io.BytesIO(imgstr)
            im = Image.open(tempimg)
            # print(np.array(im))
            # im.show()
            return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))
    else:
        return render(request, "mainpage.html")
