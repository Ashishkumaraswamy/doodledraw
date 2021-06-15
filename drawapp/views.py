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

imgques=[]
cnt=-1
gamecat=['airplane', 'alarm clock', 'angel', 'ant', 'apple', 'arm',
       'armchair', 'ashtray', 'axe', 'backpack', 'banana', 'barn',
       'baseball bat', 'basket', 'bathtub', 'bear ', 'bed', 'bee',
       'beer-mug', 'bell', 'bench', 'bicycle', 'binoculars', 'blimp',
       'book', 'bookshelf', 'boomerang', 'bottle opener', 'bowl', 'brain',
       'bread', 'bridge', 'bulldozer', 'bus', 'bush', 'butterfly',
       'cabinet', 'cactus', 'cake', 'calculator', 'camel', 'camera',
       'candle', 'cannon', 'canoe', 'car ', 'carrot', 'castle', 'cat',
       'cell phone', 'chair', 'chandelier', 'church', 'cigarette',
       'cloud', 'comb', 'computer monitor', 'computer-mouse', 'couch',
       'cow', 'crab', 'crane ', 'crocodile', 'crown', 'cup', 'diamond',
       'dog', 'dolphin', 'donut', 'door', 'door handle', 'dragon', 'duck',
       'ear', 'elephant', 'envelope', 'eye', 'eyeglasses', 'face', 'fan',
       'feather', 'fire hydrant', 'fish', 'flashlight', 'floor lamp',
       'flower with stem', 'flying bird', 'flying saucer', 'foot', 'fork',
       'frog', 'frying-pan', 'giraffe', 'grapes', 'grenade', 'guitar',
       'hamburger', 'hammer', 'hand', 'harp', 'hat', 'head',
       'head-phones', 'hedgehog', 'helicopter', 'helmet', 'horse',
       'hot air balloon', 'hot-dog', 'hourglass', 'house',
       'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo', 'key',
       'keyboard', 'knife', 'ladder', 'laptop', 'leaf', 'lightbulb',
       'lighter', 'lion', 'lobster', 'loudspeaker', 'mailbox',
       'megaphone', 'mermaid', 'microphone', 'microscope', 'monkey',
       'moon', 'mosquito', 'motorbike', 'mouse ', 'mouth', 'mug',
       'mushroom', 'nose', 'octopus', 'owl', 'palm tree', 'panda',
       'paper clip', 'parachute', 'parking meter', 'parrot', 'pear',
       'pen', 'penguin', 'person sitting', 'person walking', 'piano',
       'pickup truck', 'pig', 'pigeon', 'pineapple', 'pipe ', 'pizza',
       'potted plant', 'power outlet', 'present', 'pretzel', 'pumpkin',
       'purse', 'rabbit', 'race car', 'radio', 'rainbow', 'revolver',
       'rifle', 'rollerblades', 'rooster', 'sailboat', 'santa claus',
       'satellite', 'satellite dish', 'saxophone', 'scissors', 'scorpion',
       'screwdriver', 'sea turtle', 'seagull', 'shark', 'sheep', 'ship',
       'shoe', 'shovel', 'skateboard', 'skull', 'skyscraper', 'snail',
       'snake', 'snowboard', 'snowman', 'socks', 'space shuttle',
       'speed-boat', 'spider', 'sponge bob', 'spoon', 'squirrel',
       'standing bird', 'stapler', 'strawberry', 'streetlight',
       'submarine', 'suitcase', 'sun', 'suv', 'swan', 'sword', 'syringe',
       't-shirt', 'table', 'tablelamp', 'teacup', 'teapot', 'teddy-bear',
       'telephone', 'tennis-racket', 'tent', 'tiger', 'tire', 'toilet',
       'tomato', 'tooth', 'toothbrush', 'tractor', 'traffic light',
       'train', 'tree', 'trombone', 'trousers', 'truck', 'trumpet', 'tv',
       'umbrella', 'van', 'vase', 'violin', 'walkie talkie', 'wheel',
       'wheelbarrow', 'windmill', 'wine-bottle', 'wineglass',
       'wrist-watch', 'zebra']

def random():
    global gamecat
    global imgques
    global cnt
    ques=np.random.choice(gamecat)
    if ques in imgques:
        random()
    else:
        ques=ques.upper()
        imgques.append(ques)
        cnt+=1




def index(request):
    return render(request, "home.html")


def game(request):
    global cnt
    global imgques
    random()
    if cnt<6:
        return render(request, "mainpage.html",{
            'imgques':imgques[cnt],
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
        # print(np.array(im))
        # im.show()
        print(imgques[cnt])
        return HttpResponse("Ball")