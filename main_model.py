"""## Imports"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scipy.io
import glob
from scipy.io import loadmat
import scipy
import pandas as pd
import io
import numpy as np
import cv2
import json
import os
import os.path
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import math
# from skimage import io, transform
from collections import namedtuple
from torchvision.utils import save_image
from PIL import Image
from matplotlib import cm
import shutil
import codecs
import timeit
from slot_functions import visualize_slot
from slot_functions import *
import time
import pylab as pl
from IPython import display
import pickle
import random

"""## ConvNet class
###our convolution class contain the architecture of our cnn neural network, it has forward method to calculate the output forward value.
"""


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):

    def __init__(self):
        dropout_prob = 0.0
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer91 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer101 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer13 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer121 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer131 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer14 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer15 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer151 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer161 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer152 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer162 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), nn.Dropout(p = dropout_prob))
        self.layer17 = nn.Sequential(
            nn.Conv2d(1024, 6, kernel_size = 1, stride = 1, padding = 0),
            nn.Identity())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer91(out)
        out = self.layer101(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer121(out)
        out = self.layer131(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer151(out)
        out = self.layer161(out)
        out = self.layer152(out)
        out = self.layer162(out)
        out = self.layer17(out)
        out = out.reshape(out.size(0), -1)

        return out


"""## Design of our input label:

*  Marking point tuple represents the coordinates of a point and coordinates of its shape and the shape of the slot line.
*   slot tuple reprents the two points, the angle between them and the shape of parking whether its parallel or perpendicular

"""


# function to calculate angle between two points
def calc_angle_2(p1, p2):
    y_diff = p2[1] - p1[1]
    x_diff = p2[0] - p1[0]
    theta = np.zeros_like(x_diff)
    theta = np.arctan2(y_diff, x_diff)

    return theta


"""### Mapper
* Maps the predicted output into : confidence , x (from 0 to 512) , y (from 0 to 512) , direction-x (from 0 to 512) , direction-y (from 0 to 512) , shape of marking point

"""


def mapper(prediction_in):
    prediction = torch.clone(prediction_in)
    for i in range(16):
        for j in range(16):
            prediction[:, 1, i, j] = prediction[:, 1, i, j] + 16 + (32 * i)
            prediction[:, 2, i, j] = prediction[:, 2, i, j] + 16 + (32 * j)

            cos_value = prediction[:, 3, i, j] / 16
            sin_value = prediction[:, 4, i, j] / 16
            # direction = math.atan(sin_value/cos_value) # if we want to know the angle
            x_val_mapped = prediction[:, 1, i, j]
            y_val_mapped = prediction[:, 2, i, j]
            x_dir = x_val_mapped + (40 * cos_value)
            y_dir = y_val_mapped + (40 * sin_value)
            prediction[:, 3, i, j] = x_dir
            prediction[:, 4, i, j] = y_dir

    return prediction


"""## Prediction"""


def predict(image, MP_model):
    device = torch.device("cuda")
    MP_model = MP_model.eval()
    out = MP_model(image).to(device)
    out = out.reshape((-1, 6, 16, 16))
    out = mapper(out)
    return out


"""## Visualizer 2"""


def visualize_after_thres(image, prediction):
    pre = prediction
    plt.imshow(image)
    for i in range(len(pre)):
        p0_x = (pre[i, 1].item())
        p0_y = (pre[i][2].item())
        p1_x = pre[i, 3].item()
        p1_y = pre[i][4].item()
        confidence = pre[i][0].item()

        angle = calc_angle([p0_x, p0_y], [p1_x, p1_y]) * (math.pi/180)
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)

        p1_x = p0_x + 50 * cos_val
        p1_y = p0_y + 50 * sin_val

        p2_x = p0_x - 50 * sin_val
        p2_y = p0_y + 50 * cos_val

        p3_x = p0_x + 50 * sin_val
        p3_y = p0_y - 50 * cos_val

        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))

        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))

        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))

        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))

        # cv2.circle(image, (p0_x,p0_y), 3, (255, 0, 0), -1)
        # cv2.circle(image, (int(pre[i, 3].item()), int(pre[i][4].item())), 3, (0, 0, 255), -1)

        cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv2.putText(image, str(confidence / 100), (p0_x, p0_y),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        if pre[i, 5].item() > 50:
            cv2.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        else:
            p3_x = int(round(p3_x))
            p3_y = int(round(p3_y))
            cv2.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)
    # cv2.imshow("image", image)
    return image


"""## Remove points lower than Thresthold 

* takes the predicted point and the threshold, check if the confidence value is less than this threshold then we remove this point



"""


def get_predicted_points(prediction, thresh):  # prediction (training,6,16,16)
    """Get marking points from one predicted feature map."""
    assert isinstance(prediction, torch.Tensor)
    if len(prediction.shape) == 3:
        prediction = prediction.reshape((1, 6, 16, 16))
    num_training_examples = prediction.shape[0]
    predicted_points = []
    prediction = prediction.detach().cpu()

    index = (torch.greater_equal(prediction[:, 0, :, :], thresh))

    prediction = prediction.permute(0, 2, 3, 1)

    for i in range(num_training_examples):
        predicted_points.append(prediction[i, index[i, :, :]])

    predicted_points_copy = []
    for i in range(len(predicted_points)):
        predicted_points_copy.append(torch.clone(predicted_points[i]))

    for i in range(len(predicted_points)):  # 3la el training examples
        for j in range(len(predicted_points[i])):  # 3la el points
            if predicted_points[i][j][1] < 10 and predicted_points[i][j][
                2] < 10:  # remove points with negative x , y values
                predicted_points_copy[i] = torch.cat(
                    [predicted_points_copy[i][0:j, :], predicted_points_copy[i][j + 1:, :]])
    return (predicted_points_copy)


"""## Remove Row from Tensor"""


def remove_row_ls(tens, row_index_ls):
    tens_copy = torch.clone(tens)
    ls = []
    for i in range(tens_copy.shape[0]):
        ls.append(i)
    ls = torch.tensor(ls)
    # index = []
    for i in range(len(row_index_ls)):
        index = ls[ls != row_index_ls[i]]
        ls = index
    index = torch.tensor(index)
    return torch.index_select(tens_copy, 0, index)


"""## Non-max suppression for near points


*   take predicted points and check if there is more than one point close to each other it will remove all of the except the one which have the highest confidence



"""


def non_maximum_suppression(pred_points):  # pred_points (training,num_points,6) (list)
    """Perform non-maxmum suppression on marking points."""
    pred_copy = pred_points
    threshold_near = 40  # expermintal
    for k, tens in enumerate(pred_copy):
        index_arr = []
        for i in range(tens.shape[0]):
            for j in range(i + 1, tens.shape[0]):
                i_x = tens[i][1]
                i_y = tens[i][2]
                j_x = tens[j][1]
                j_y = tens[j][2]
                abs_x = abs(j_x - i_x)
                abs_y = abs(j_y - i_y)
                abs_dis = math.sqrt(math.pow(abs_x, 2) + math.pow(abs_y, 2))
                if abs_dis <= threshold_near:
                    idx = i if tens[i][0] < tens[j][0] else j
                    index_arr.append(idx)

        if (len(index_arr) != 0):
            pred_copy[k] = remove_row_ls(tens, index_arr)
    return pred_copy



def calc_angle(p1, p2):
    y_diff = p2[1] - p1[1]
    x_diff = p2[0] - p1[0]
    if x_diff == 0 and y_diff == 0:
        return 0
    theta = math.atan(abs(y_diff / x_diff)) * (180 / math.pi)
    if x_diff >= 0 and y_diff >= 0:
        return theta
    elif x_diff < 0 and y_diff >= 0:
        return 180 - theta
    elif x_diff < 0 and y_diff < 0:
        return - 180 + theta
    elif x_diff > 0 and y_diff < 0:
        return -1 * theta
    else:
        return theta


"""### initialize marking points predictor"""


def init_marking_points_model():
    device = torch.device("cuda")
    model = ConvNet().to(device)
    checkpoint = torch.load(r'model_weights')  # weights path
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


"""### Takes an image returns predicted Marking points"""


def image_predict_marking_points(input_image, MP_model):
    device = torch.device("cuda")
    image_transform = ToTensor()
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image_2 = image_transform(input_image)

    # if image is not 512 x 512
    # input_image_2 = input_image_2.permute(1,2,0)
    # input_image_2 = transform.resize(input_image_2, (512, 512))
    # input_image_2 = torch.tensor(input_image_2).permute(2,0,1)

    predict_awal = predict(input_image_2.reshape((1, 3, 512, 512)).to(device), MP_model)
    predict_ba3d = get_predicted_points(predict_awal, 70)
    predict2 = non_maximum_suppression(predict_ba3d)
    output_image = visualize_after_thres(input_image, predict2[0])
    return predict2 , output_image
