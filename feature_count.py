import pandas as pd 
import os
import csv
from read_write_csv import get_csv_data ,write_csv_data, write_csv_data_test
feature_count = {}
label_path = "groundturth.csv"       # 車體的實際值csv檔
label = get_csv_data(label_path)     # 讀取實際值資料
sub_train_path = r'dataset\B18_dataset-5\train'
sub_val_path = r'dataset\B18_dataset-5\valid'
from pathlib import Path


# 計算從原圖擷取出的特徵圖數量
for sub_path in [sub_train_path, sub_val_path]:
        sub_original_path = os.path.join(sub_path, "original")
        sub_feature_path = os.path.join(sub_path, "feature")
        # 產生原圖的實際值對照表
        new_label = []
        for file in os.listdir(sub_original_path):
            file_name = Path(file).stem                  # 從路徑中取得 檔名
            file_carnumber = file_name.split("_")[0]     # 從檔名中取得 車號
            file_cardoorplace =  file_name.split("_")[3] # 從檔名中取得 車門位置
            for y in label:                      
                if file_carnumber == y[0]:         # 從所有的label 中取得該車號的資料
                    if file_cardoorplace == "front":
                        new_label.append([file] + y[4:8])
                    else:
                        new_label.append([file] + y[8:12])
        save_o_path = sub_path + "_original_groundturth.csv"
        write_csv_data(new_label, save_o_path)
        # 產生特徵圖的實際值對照表
        feature_count = {}          # 記錄每張圖片可以擷取的特徵圖數量
        new_feature_label = []
        for file in os.listdir(sub_feature_path):
            file_name = Path(file).stem                  # 從路徑中取得 檔名
            file_carnumber = file_name.split("_")[0]     # 從檔名中取得 車號
            file_cardoorplace =  file_name.split("_")[3] # 從檔名中取得 車門位置
            for y in label:
                if file_carnumber == y[0]:         # 從所有的label 中取得該車號的資料，
                    if file_cardoorplace == "front":
                        new_feature_label.append([file] + y[4:8])
                    else:
                        new_feature_label.append([file] + y[8:12])
                    file_num_door = file_carnumber +'_' +file_cardoorplace
                    feature_count[file_num_door]=feature_count.get(file_num_door, 0)+1  # 計算原圖擷取的特徵圖數量，前後門分開計算
        save_f_path = sub_path + "_feature_groundturth.csv"
        write_csv_data(new_feature_label, save_f_path)

        count = []  # 要寫入csv，各原圖可擷取的特徵圖數量
        df = pd.read_csv(save_o_path)    # 讀取原圖的csv ，根據檔名加入特徵圖個數
        for filename in df['filename']:
            name = Path(filename).stem
            file_carnumber = name.split("_")[0]     # 從檔名中取得 車號
            file_cardoorplace =  name.split("_")[3] # 從檔名中取得 車門位置
            file_num_door = file_carnumber + '_' + file_cardoorplace
            try :
                count.append(feature_count[file_num_door])   # 有特徵圖
            except:
                count.append('0')                            # 沒有擷取到特徵圖
        df['count'] = count
        df.to_csv(save_o_path,index=None)
        





