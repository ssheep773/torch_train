import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# from skimage.io import imread, imshow
# from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
# from skimage import color, exposure, transform
# from skimage.exposure import equalize_hist


# result = cv2.dft(np.float32(src), flags=cv2.DFT_COMPLEX_OUTPUT)
# # 將頻譜低頻從左上角移動至中心位置
# dft_shift = np.fft.fftshift(result)
# # 頻譜影象雙連結複數轉換為 0-255 區間
# result1 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


def myfft(name):
    src = cv2.imread(name)
    f = np.fft.fft2(src, (512, 512))
    fshift = np.fft.fftshift(f)
    result1 = 20 * np.log(np.abs(fshift))
    cv2.imwrite(name, result1)


for set in os.listdir("datasetfftrgb"):
    print(set)
    for folder in ["train", "valid", "test"]:
        set_path = os.path.join("datasetfftrgb", set, folder, "feature")
        # print(set_path)
        for file in os.listdir(set_path):
            path = os.path.join(set_path, file)
            print(path)
            myfft(path)
