import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def myfft(name):
    src = cv2.imread(name)
    result1 = cv2.resize(src, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite(name, result1)


for set in os.listdir("datasetlowpass"):
    print(set)
    for folder in ["train", "valid", "test"]:
        set_path = os.path.join("datasetlowpass", set, folder, "feature")
        # print(set_path)
        for file in os.listdir(set_path):
            path = os.path.join(set_path, file)
            print(path)
            myfft(path)
