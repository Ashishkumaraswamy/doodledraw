from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import RequestContext
from PIL import Image, ImageOps, ImageCms
import re
import io
import base64
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
import warnings
import cv2 as cv2
warnings.filterwarnings('ignore')

# Create your views here.

ans = []
imgques = []
gamecat = ['spider', 'bed', 'sock', 'frying_pan', 'grapes', 'basketball', 'axe', 'wristwatch', 'bread', 'anvil',
           'mountain', 'rifle', 'rainbow', 'stop_sign', 'power_outlet', 'alarm_clock',
           'drums', 'lollipop', 'cookie', 'knife', 'scissors', 'flower', 'pencil', 'apple', 'car', 'tent', 'cat', 'beard',
           'umbrella', 'butterfly', 'radio', 'shovel', 'sun', 'syringe', 'bird', 'sword', 'book', 'face', 'baseball', 'laptop', 'hammer',
           'ice_cream', 'spoon', 'tree', 'microphone', 'bridge', 'traffic_light', 'star', 'diving_board', 'shorts',
           'chair', 'eyeglasses', 'fan', 'tooth', 'cell_phone', 'headphones', 'saw', 'pillow', 'cup', 'square', 'circle',
           'light_bulb', 'paper_clip', 'screwdriver', 'tennis_racquet', 'coffee_cup', 'envelope', 'hat', 'hot_dog', 'ceiling_fan',
           'suitcase', 'bench', 'moon', 'wheel', 'cloud', 'eye', 'line', 'pants', 'airplane', 'smiley_face', 'camera', 'moustache',
           'pizza', 'triangle', 'broom', 'key', 'bicycle', 'snake', 'donut', 'clock', 'dumbbell', 'candle', 'ladder', 't-shirt', 'mushroom',
           'helmet', 'baseball_bat', 'lightning', 'table', 'door']
for i in range(len(gamecat)):
    gamecat[i] = gamecat[i].replace('_', ' ')


def normalize(data):
    "Takes a list or a list of lists and returns its normalized form"
    return np.interp(data, [0, 255], [-1, 1])


def random(request):
    global gamecat
    global imgques

    ques = np.random.choice(gamecat)
    if ques in imgques:
        random(request)
    else:
        ques = ques.upper()
        request.session["imgques"].append(ques)
        request.session["cnt"] += 1


def index(request):
    global ans
    ans = []
    if "cnt" not in request.session:
        request.session["cnt"] = -1
    if "imgques" not in request.session:
        request.session["imgques"] = imgques
    if "ans" not in request.session:
        request.session["ans"] = ans
    if request.session['cnt'] < 6:
        request.session["cnt"] = -1
        request.session["imgques"] = []
    request.session["cnt"] = -1
    request.session["imgques"] = imgques
    request.session["ans"] = ans
    return render(request, "home1.html")


def game(request):
    global imgques
    return render(request, "mainpage.html", {
        'imgques': request.session["imgques"][-1],
    })


def result(request):
    return render(request, "result.html")


def question(request):
    global imgques
    global ans
    if "cnt" not in request.session:
        request.session["cnt"] = -1
    if "imgques" not in request.session:
        request.session["imgques"] = imgques
    random(request)
    if request.session['cnt'] < 6:
        ans.append(0)
        return render(request, "question.html", {
            'imgques': request.session["imgques"][-1],
            'count': request.session["cnt"]+1,
        })
    else:
        request.session['ans'].extend(ans)
        print(request.session["ans"])
        return render(request, "result.html", {
            'imgques': request.session["imgques"],
            'ans': request.session["ans"],
            'score': np.sum(ans),
        })


def get_canvas(request):
    global imgques
    global ans
    if request.method == "POST":
        captured_image = request.POST['canvas_data']
        imgstr = re.search('base64,(.*)', captured_image).group(1)
        imgstr = base64.b64decode(imgstr)
        tempimg = io.BytesIO(imgstr)
        im = Image.open(tempimg)
        im = im.convert('RGB')
        im.save('drawapp\static\drawapp\\temp' +
                str(request.session['cnt']+1)+'.jpg')
        im = Image.open('drawapp\static\drawapp\\temp' +
                        str(request.session['cnt']+1)+'.jpg')
        cat = classify(im)
        print(cat)
        if cat == request.session["imgques"][-1].lower():
            ans[-1] = 1
            print("Ans", ans)
            return HttpResponse('Oh! I got it. It\'s a '+cat)
        return HttpResponse('I guess '+cat)


def classify(image):
    cat = predictimage(image)
    return cat


def predictimage(im):
    model = tf.keras.models.load_model(os.path.join("./drawapp/", "keras.h5"))
    image_size = 28
    imgcrop = inputpreprocessing(im)
    if imgcrop.size == 0:
        return '....'
    x = imgcrop
    x = x.reshape(image_size, image_size, 1).astype('float32')
    x = x*3
    pred = model.predict(np.expand_dims(x, axis=0))[0]
    ind = (-pred).argsort()[:5]
    latex = [gamecat[i] for i in ind]
    return latex[0]


# def cropimage(im):
#     im = ImageOps.grayscale(im)
#     im = np.array(im)
#     im = np.where(im == 255, 0, im)
#     im = np.where(im != 0, 255-im, im)
#     imagedim = []
#     start = []
#     maxwidth = 0
#     startrow = -1
#     endrow = -1
#     for i in range(im.shape[0]):
#         listrow = []
#         for j in range(im.shape[1]):
#             if im[i][j] != 0:
#                 listrow = im[i, j:].tolist()
#                 start.append(j)
#                 break
#         for k in range(len(listrow)-1, 0, -1):
#             if listrow[k] != 0:
#                 listrow = listrow[:k+1]
#                 if len(listrow) != 0:
#                     if start[-1]+len(listrow) > maxwidth:
#                         maxwidth = start[-1]+len(listrow)
#                     if len(imagedim) == 0:
#                         startrow = i
#                     endrow = i
#                     imagedim.append(listrow)
#                 break
#     imgcrop = np.array([])
#     try:
#         min_start = np.min(start)
#         imgcrop = im[startrow-50:endrow+50, min_start-50:maxwidth+50]
#     except ValueError as e:
#         pass
#     return imgcrop


def printarray2d(x):
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            print(np.round(x[1][j], 3), end=" ")
        print()


def printarray(x):
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            print(np.round(x[i][j][0], 3), end=" ")
        print()


def inputpreprocessing(img):

    # img = "/content/traffic1.jpeg"

    # img = Image.open(img)
    # img=np.array(img)
    if img.mode == "CMYK":
        # color profiles can be found at C:\Program Files (x86)\Common Files\Adobe\Color\Profiles\Recommended
        img = ImageCms.profileToProfile(
            img, "USWebCoatedSWOP.icc", "sRGB_Color_Space_Profile.icm", outputMode="RGB")
    # PIL image -> OpenCV image; see https://stackoverflow.com/q/14134892/2202732
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # (4) Crop and save it
    x, y, w, h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]

    # add border/padding around the cropped image
    # dst = cv2.copyMakeBorder(dst, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])

    # cv2.imshow("image", dst)
    # plt.imshow(dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # create/write to file
    # cv2.imwrite("001.png", dst)
    WHITE = [255, 255, 255]
    im = cv2.copyMakeBorder(dst.copy(), 50, 50, 50, 50,
                            cv2.BORDER_CONSTANT, value=WHITE)
    img = np.array(im)
    img = np.where(img == 255, 0, img)
    im = np.where(img != 0, 255-img, img)
    # im=Image.fromarray(img)
    im = Image.fromarray(im)
    im = ImageOps.grayscale(im)
    im = np.array(im)
    im = resize(im, (28, 28))
    # plt.imshow(im)
    return im
