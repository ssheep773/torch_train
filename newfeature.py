# trainning 資料前處理

# 前後車門總圖 : 將5張前門或是後門和在一起方便傳輸
# 0. 選取要做train 的圖片，可能是只取白色
# 1.將 前後車門總圖 分為train, val
# 2.將分好後的train, val 做特徵擷取
# 3.為特徵擷取後的圖做label ，依照他的檔名知道是哪一台車的特徵圖，存在csv
# 4. trainning
import os
import cv2
import tqdm
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import math


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


# 特徵擷取
def new_feature_selection(img, filename, Cut_place=0, Perimeter=8000, window_size=256):
    img = img[Cut_place:, :2448]  # C12 front
    # img = cv2.bilateralFilter(img, 23, 50, 50)  # 去雜訊
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色階轉換

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # n_img = clahe.apply(gray)
    # img = n_img

    ret, dst = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)  # 二值化
    kernel = np.ones((15, 15), np.uint8)  # 侵蝕
    dst = cv2.erode(dst, kernel, iterations=1)
    # x = cv2.Sobel(dst, cv2.CV_16S, 1, 0)# 邊緣偵測
    # y = cv2.Sobel(dst, cv2.CV_16S, 0, 1)
    # absX = cv2.convertScaleAbs(x)# 轉回uint8
    # absY = cv2.convertScaleAbs(y)
    # dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # # kernel1 = np.ones((7,7), np.uint8)   #膨脹
    # # dst1 = cv2.dilate(dst1, kernel1, iterations = 2)
    # dst1[dst1>60] = 255
    # img = n_img - 0.4*dst1
    # img = n_img - 0.4*dst

    counter, hierarcy = cv2.findContours(
        dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # 找輪廓
    feature_count = 0

    for i in counter:
        if cv2.contourArea(i) > Perimeter:
            M = cv2.moments(i)  # 計算x ,y 的中心點
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])

            leftmost = tuple(i[i[:, :, 0].argmin()][0])
            rightmost = tuple(i[i[:, :, 0].argmax()][0])

            contour_size = [
                abs(leftmost[0] - rightmost[0]),
                abs(leftmost[1] - rightmost[1]),
            ]  # window size 隨特徵的大小縮放

            if min(contour_size) > 100:  # 去掉過小的特徵 ，size過小的是不需要的特徵
                new_window_size = window_size  # window_size等於預設值
                if (
                    y + new_window_size > img.shape[0]
                    or x + new_window_size > img.shape[1]
                    or x - new_window_size < 0
                    or y - new_window_size < 0
                ):  # 若會切到圖片邊界 ，則根據contour_size縮小window_size
                    if contour_size[0] < new_window_size:
                        new_window_size = contour_size[0]
                    if contour_size[1] < new_window_size:
                        new_window_size = contour_size[1]
                    # print("cut border with new size :", new_window_size)

                window = img[
                    y - new_window_size : y + new_window_size,
                    x - new_window_size : x + new_window_size,
                ]
                # if y + new_window_size > img.shape[0] or x + new_window_size > img.shape[1] or x - new_window_size < 0 or y - new_window_size < 0 :
                #     continue

                feature_name = filename + "_feature-" + str(feature_count) + ".jpg"
                # print(feature_name)
                try:
                    # print(feature_name)
                    # window = cv2.resize(window, (512, 512), interpolation=cv2.INTER_AREA)
                    # window = cv2.bilateralFilter(window, 23, 50, 50)#去雜訊
                    # window = img_pad(window, fixed_size=window_size*2, type1='side')
                    cv2.imwrite(feature_name, window)
                    feature_count += 1
                    # print(feature_name)
                    # start_point = (x-new_window_size, y-new_window_size)
                    # end_point = (x+new_window_size, y+new_window_size)
                    # img = cv2.rectangle(img, start_point, end_point,  (0, 0, 255) , 3)
                except:
                    # print("GG")
                    continue
                # feature_count += 1

                # cv2.imwrite(f"{name}_{feature_count}.png", img)


if __name__ == "__main__":
    for img in os.listdir("img"):
        name = img[:-4]
        img_path = os.path.join("img", img)
        img = cv2.imread(img_path)
        print(img_path)
        new_feature_selection(img, name)
