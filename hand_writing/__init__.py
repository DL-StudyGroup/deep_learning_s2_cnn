# coding: utf-8
# import util
import torch.nn as nn
import torch
import os
import numpy as np
import PIL.ImageOps


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                padding=2,  # if stride=1, padding=(kernel_size - 1) /2 = (5 -1)/2
            ),  # -> (16, 28, 28)
            nn.ReLU(),  # -> (16, 28, 28)
            nn.MaxPool2d(kernel_size=2),  # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # -> (32, 14, 14)
            nn.ReLU(),  # -> (32, 14, 14)
            nn.MaxPool2d(2)  # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output


def pred_matrix(img_data, param_path):
    img_matrix = torch.FloatTensor(img_data)
    img_matrix = np.reshape(img_matrix, (1, 1, 28, 28))
    cnn_params = torch.load(param_path)
    new_cnn = CNN()
    new_cnn.load_state_dict(cnn_params)
    new_cnn.eval()
    output = new_cnn(img_matrix)
    pred_y = torch.max(output, 1)[1].numpy()
    print(pred_y)
    return pred_y
