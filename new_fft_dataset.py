import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def myfft(name):
    img = cv2.imread(name)

    R_channel = img[:, :, 0]
    G_channel = img[:, :, 1]
    B_channel = img[:, :, 2]

    # 对每个通道分别执行FFT
    R_fft = np.fft.fft2(R_channel, (512, 512), axes=(0, 1), norm="backward")
    G_fft = np.fft.fft2(G_channel, (512, 512), axes=(0, 1), norm="backward")
    B_fft = np.fft.fft2(B_channel, (512, 512), axes=(0, 1), norm="backward")

    R_fshift = np.fft.fftshift(R_fft)
    G_fshift = np.fft.fftshift(G_fft)
    B_fshift = np.fft.fftshift(B_fft)
    restored_fft = np.stack((R_fshift, G_fshift, B_fshift), axis=-1).real
    cv2.imwrite(name, restored_fft)


for set in os.listdir("datasetfftrgb"):
    print(set)
    for folder in ["train", "valid", "test"]:
        set_path = os.path.join("datasetfftrgb", set, folder, "feature")
        # print(set_path)
        for file in os.listdir(set_path):
            path = os.path.join(set_path, file)
            print(path)
            myfft(path)
