import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
# from skimage.color import rgb2lab, deltaE_cie76
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import pickle
import time


def get_hist(img):
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    hist = np.stack((np.squeeze(hist_r), np.squeeze(hist_g), np.squeeze(hist_b)), axis = 1)
    return hist


def calc_hist_ls(path):
    images_names = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            images_names.append(os.path.splitext(file)[0])
    hist_ls = []
    for i in range(len(images_names)):
        image = cv2.imread(f"{path}/{images_names[i]}.jpg")
        image_masked = get_handi_masked(image)
        hist = get_hist(image_masked)[5:, :]
        hist_ls.append(hist)
    return hist_ls


def get_mean(img):
    mean = np.mean(img, axis = (0, 1))
    return mean


def compare_hist(img1, img2, threshold = 0.7):
    hist1 = get_hist(img1)[5:, :]
    hist2 = get_hist(img2)[5:, :]
    m = np.minimum(hist1, hist2) / (np.sum(hist2))
    s = np.sum(m)
    print(s)
    if s > threshold:  # similair
        return True
    else:
        return False


def compare_hist_list(img, hist_ls = [], threshold = 0.3, match_threshold = 0.2):  # Tune match threshold
    # image is bgr restrictly !!!!!!
    if len(hist_ls) == 0:
        open_file = open("handicapped_ls.pkl", "rb")
        hist_ls = pickle.load(open_file)
        open_file.close()

    img = get_handi_masked(img)
    hist_test = get_hist(img)[5:, :]
    accum_flags = 0
    for i in range(len(hist_ls)):
        m = np.minimum(hist_test, hist_ls[i]) / (np.sum(hist_ls[i]) + 1)
        s = np.sum(m)
        if s > threshold:  # similair
            accum_flags += 1
    match_ratio = accum_flags / len(hist_ls)
    # print(match_ratio)
    if match_ratio > match_threshold:
        return True
    else:
        return False


def get_handi_masked(image):  # image is bgr restrictly !!!!!!
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    offset = 80 - np.mean(image_hsv[:, :, 2])  # 80
    image_bright = np.copy(image_hsv)
    image_bright[:, :, 2] = image_bright[:, :, 2] + offset
    lower_blue = np.array([100, 100, 50])  # 100 #190,50,30
    upper_blue = np.array([255, 255, 255])  # 255 # 270,100,100
    mask = cv2.inRange(image_bright, lower_blue, upper_blue)
    result_hsv = cv2.bitwise_and(image_bright, image_bright, mask = mask)
    image_rgb_res = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)
    return image_rgb_res


# im_path = r''  # image path
# image = cv2.imread(im_path)  # do not convert to rgb!!
# similar = compare_hist_list(image)
# print(similar)
