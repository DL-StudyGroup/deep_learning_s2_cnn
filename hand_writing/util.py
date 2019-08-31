# coding: utf-8

import torch
from PIL import Image
import PIL.ImageOps
import numpy as np


def test():
    img2matrix("./data/8.png")


def img2matrix(file_name):
    im = Image.open(file_name)
    # im.show()
    width, height = im.size
    im = im.convert("L")
    im = PIL.ImageOps.invert(im)
    data = np.array(im)
    tensor = torch.FloatTensor(data) / 255.0
    # data = im.getdata()
    # data = np.matrix(data, dtype='float') / 255.0
    new_data = np.reshape(tensor, (1, 1, height, width))

    print(new_data.shape)
    return new_data
