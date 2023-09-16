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
# from bilateralFilter_torch import bilateralFilter
# from read_write_csv import get_csv_data
# from read_write_csv import write_csv_data


# 特徵擷取
def feature_selection(img, filename, Cut_place=0, Perimeter=8000, window_size=256):
    # img = img[500:,:] #P15 front
    img = img[Cut_place:,:2448] #C12 front
    # cv2.imwrite(f"split.jpg",img)

    img = cv2.bilateralFilter(img, 23, 20, 20)#去雜訊
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 色階轉換
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    n_img = clahe.apply(gray)
    img = n_img
    # cv2.imwrite(f"clahe.jpg",n_img)
    
    ret , thres = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)# 二值化
    # cv2.imwrite("threshold.png", thres)
    kernel = np.ones((5,5), np.uint8)     #侵蝕
    dst = cv2.erode(thres, kernel, iterations = 1)

    x = cv2.Sobel(dst, cv2.CV_16S, 1, 0)# 邊緣偵測
    y = cv2.Sobel(dst, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)# 轉回uint8
    absY = cv2.convertScaleAbs(y)
    dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # kernel1 = np.ones((3,3), np.uint8)   #膨脹
    # dst1 = cv2.dilate(dst1, kernel1, iterations = 2)
    dst1[dst1>60] = 255
    

    # gray = 255- gray 
    # merage = gray*1 + dst1*0.3
    # merage = 255 - merage
    merage = n_img - 0.3*dst1
    merage = merage - dst

    # cv2.imwrite("merage.png", merage)

    counter , hierarcy = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # 找輪廓
    feature_count = 0
    for i in counter:
        if cv2.contourArea(i) > Perimeter:
            M = cv2.moments(i)             # 計算x ,y 的中心點
            x = int( M['m10'] / M['m00'] )  
            y = int( M['m01'] / M['m00'] )
            leftmost = tuple(i[i[:,:,0].argmin()][0])
            rightmost = tuple(i[i[:,:,0].argmax()][0])
            f_size = [abs(leftmost[0] - rightmost[0]), abs(leftmost[1] - rightmost[1])]  # window size 隨特徵的大小縮放

            if min(f_size) < 100:                # 去掉過小的特徵 ，size過小的輪廓是不需要的特徵
                print(f"contourArea with size {min(f_size)}too small del")
            else:
                cur_window_size = window_size
                if y + cur_window_size > img.shape[0] or x + cur_window_size > img.shape[1] or x - cur_window_size < 0 or y - cur_window_size < 0 :
                    if f_size[0] < cur_window_size:
                        cur_window_size = f_size[0]
                    if f_size[1] < cur_window_size:
                        cur_window_size = f_size[1]
                    print("cut border with new size :", cur_window_size)
                
                for x, y in [(x,y)]:
              
                    window = merage[ y - cur_window_size : y+cur_window_size , x-cur_window_size : x+cur_window_size ]
                    # if y + cur_window_size > img.shape[0] or x + cur_window_size > img.shape[1] or x - cur_window_size < 0 or y - cur_window_size < 0 :
                    #     continue
                        
                    print(filename + "_feature-"+str(feature_count)+".jpg")
        
                    feature_name =  filename + "_feature-"+str(feature_count)+".jpg"
                  
                    try :
                        cv2.imwrite(feature_name, window)

                        # start_point = (x-cur_window_size, y-cur_window_size)
                        # end_point = (x+cur_window_size, y+cur_window_size)
                        # img = cv2.rectangle(img, start_point, end_point,  (0, 0, 255) , 3)

                        feature_count += 1
                    except:
                        print("GG")
                    # cv2.imwrite(f"{name}_{feature_count}.png", img)


if __name__ == '__main__':
    
    # for img in os.listdir("img"):
        # img = "57266_C12_235_best.jpg"
        img = "23087_C12_235_worse.jpg"
        name = img[:-4]
        img_path = os.path.join("bestworse", img)
        img = cv2.imread(img_path)
        print(img_path)
        feature_selection(img, name)
    
   



