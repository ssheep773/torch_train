import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from PIL import Image


def img_pad(pil_file, fixed_size, type1="side"):
    # print(pil_file.shape)
    try:
        h, w, c = pil_file.shape
    except:
        h, w = pil_file.shape
    if h == 0 or w == 0:
        return None
    pad_h = fixed_size - h
    pad_w = fixed_size - w
    # print(pad_h,pad_w)
    array_file = np.array(pil_file)
    if type1 == "side":
        try:
            array_file = np.pad(
                array_file, ((0, pad_h), (0, pad_w), (0, 0)), "constant"
            )
        except:
            array_file = np.pad(array_file, ((0, pad_h), (0, pad_w)), "constant")
    else:
        array_file = np.pad(
            array_file,
            (
                (math.ceil(pad_h / 2), math.floor(pad_h / 2)),
                (math.ceil(pad_w / 2), math.floor(pad_w / 2)),
                (0, 0),
            ),
            "constant",
        )
    return array_file


def lowpass(name, filtersize):
    img = cv2.imread(name)
    fft = np.fft.fftn(img)
    fshift = np.fft.fftshift(fft)

    fshift_r = fshift[:, :, 0]
    fshift_g = fshift[:, :, 1]
    fshift_b = fshift[:, :, 2]

    # 低通濾波器
    rows, cols, _ = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    # filtersize = 30
    mask = np.zeros((rows, cols), np.uint8)
    mask[
        crow - filtersize : crow + filtersize, ccol - filtersize : ccol + filtersize
    ] = 1

    # 濾波器 乘以fft
    R_fshift_filtered = fshift_r * mask
    G_fshift_filtered = fshift_g * mask
    B_fshift_filtered = fshift_b * mask

    # 將濾波後的fft 做ifft 前需要將3個channel合併
    fft_filtered = np.stack(
        (R_fshift_filtered, G_fshift_filtered, B_fshift_filtered), axis=-1
    )

    # ifft
    ifshift = np.fft.ifftshift(fft_filtered)
    ifft_image = np.fft.ifftn(ifshift)
    ifft_image = img_pad(abs(ifft_image), fixed_size=512, type1="side")
    cv2.imwrite(name, ifft_image)


dataset = "datasetlowpass20"
filtersize = 20

for set in os.listdir(dataset):
    print(set)
    for folder in ["train", "valid", "test"]:
        set_path = os.path.join(dataset, set, folder, "feature")
        # print(set_path)
        for file in os.listdir(set_path):
            path = os.path.join(set_path, file)
            # print(path)
            lowpass(path, filtersize)
