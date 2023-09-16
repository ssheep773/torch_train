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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from read_write_csv import get_csv_data
from read_write_csv import write_csv_data
from newfeature import new_feature_selection, img_pad


# 選擇要納入訓練的車籍資料，ex: 車色，車型。 預設為全部納入訓練
def select_train_datatype(data_path, color="all", cartype="all"):
    select_color_file, select_file = [], []
    # 為資料集命名
    if color == "all" and cartype == "all":
        subdataset_name = "all_dataset"
    elif color == "all" or cartype == "all":
        subdataset_name = f"{color}{cartype}-dataset".replace("all", "")
    else:
        subdataset_name = f"{color}_{cartype}-dataset"

    # 從所有的資料集檔案中，選取符合輸入color 的車色
    for file in os.listdir(data_path):
        file_color = file.split("_")[2]

        if color == file_color:
            select_color_file.append(file)
        elif color == "all":
            select_color_file.append(file)

    # 從剛剛選取府和顏色的資料集檔案中，選取符合輸入cartype 的車型
    for file in select_color_file:
        file_cartype = file.split("_")[1]
        file = os.path.join(data_path, file)
        if cartype == file_cartype:
            select_file.append(file)
        elif cartype == "all":
            select_file.append(file)
    return subdataset_name, select_file


# 將選取的檔案加入dataset，並創建一個資料夾存放
# select_file_list 選取的資料集檔案
# datasetpath
def create_subdataset(select_file_list, datasetpath, subdataset_name, label):
    # 分 train valid
    y = np.zeros(len(select_file_list))
    kfold = StratifiedKFold(n_splits=5, random_state=18, shuffle=True).split(
        select_file_list, y
    )
    print(kfold)
    print(len(select_file_list))

    for k, (train, val) in enumerate(kfold):
        # print(k,train,val)
        sub_kfold_name = subdataset_name.replace(
            "dataset", f"kfold-{k}-{len(select_file_list)}"
        )
        # sub_kfold_name = sub_kfold_name + "hole"

        # print(sub_kfold_name)
        train_list, val_test_list = [], []
        for t in train:
            train_list.append(select_file_list[t])
        for v in val:
            val_test_list.append(select_file_list[v])
        y = np.zeros(len(val_test_list))
        val_list, test_list, Y_train, Y_test = train_test_split(
            val_test_list, y, test_size=0.5, random_state=12
        )
        print(
            f"{sub_kfold_name}  ==>  {len(train_list)} : {len(val_list)} : {len(test_list)}"
        )

        subdataset_path = os.path.join(datasetpath, sub_kfold_name)
        sub_train_path = os.path.join(subdataset_path, "train")
        sub_val_path = os.path.join(subdataset_path, "valid")
        sub_test_path = os.path.join(subdataset_path, "test")

        for path in [subdataset_path, sub_train_path, sub_val_path, sub_test_path]:
            if not os.path.isdir(path):
                os.makedirs(path)

        # 特徵擷取
        for folder, li in [
            (sub_train_path, train_list),
            (sub_val_path, val_list),
            (sub_test_path, test_list),
        ]:
            for merage_img in tqdm.tqdm(li):
                # print(merage_img)
                # train 跟 valid 都有儲存原圖的original跟儲存特徵圖的feature資料夾
                sub_original_path = os.path.join(folder, "original")
                sub_feature_path = os.path.join(folder, "feature")
                for path in [sub_original_path, sub_feature_path]:
                    if not os.path.isdir(path):
                        os.makedirs(path)
                # 複製原圖
                img_name = os.path.split(merage_img)[1]
                original_path = os.path.join(sub_original_path, img_name)
                shutil.copy(merage_img, original_path)
                # 擷取特徵圖的前處理
                im = cv2.imread(merage_img)
                h, w, c = im.shape
                split_time = w // 2448
                for i in range(
                    split_time
                ):  # 因為原圖是5張圖concate在一起 所以要先分開，再擷取特徵，沒有分開不易擷取到特徵
                    im_split = im[:, i * 2448 : (i + 1) * 2448]
                    new_name = merage_img[:-4] + "_image-" + str(i)
                    new_name = os.path.split(new_name)[-1]  # 把原圖存取的資料夾路徑去掉
                    new_path = os.path.join(sub_feature_path, new_name)  # 把目的 資料夾位置 加進去
                    new_feature_selection(
                        im_split, new_path, Cut_place=0, Perimeter=8000, window_size=256
                    )  # 選取特徵後 儲存特徵

        # 建立特徵擷取後的資料集 groundturth 分為 train 跟 val
        for sub_path in [sub_train_path, sub_val_path, sub_test_path]:
            sub_original_path = os.path.join(sub_path, "original")
            sub_feature_path = os.path.join(sub_path, "feature")
            # 產生原圖的實際值對照表
            new_label = []
            for file in os.listdir(sub_original_path):
                file_name = Path(file).stem
                file_carnumber = file_name.split("_")[0]  # 取得 車號
                file_cardoorplace = file_name.split("_")[3]  # 取得 車門位置
                for y in label:
                    if file_carnumber == y[0]:  # 從所有的label 中取得該車號的資料
                        if file_cardoorplace == "front":
                            new_label.append([file] + y[4:8])
                        else:
                            new_label.append([file] + y[8:12])
            save_o_path = sub_path + "_original_groundturth.csv"
            write_csv_data(new_label, save_o_path)

            # 產生特徵圖的實際值對照表
            feature_count = {}  # 記錄每張圖片可以擷取的特徵圖數量
            new_feature_label = []
            for file in os.listdir(sub_feature_path):
                file_name = Path(file).stem
                file_carnumber = file_name.split("_")[0]  # 取得 車號
                file_cardoorplace = file_name.split("_")[3]  # 取得 車門位置
                for y in label:
                    if (
                        file_carnumber == y[0]
                    ):  # 從所有的label 中取得該車號的資料，不選擇從new_label抓取是因為格式問題，這樣抓不用更改方法，只是時間較久但差異不大
                        if file_cardoorplace == "front":
                            new_feature_label.append([file] + y[4:8])
                        else:
                            new_feature_label.append([file] + y[8:12])
                        file_num_door = file_carnumber + "_" + file_cardoorplace
                        feature_count[file_num_door] = (
                            feature_count.get(file_num_door, 0) + 1
                        )  # 計算原圖擷取的特徵圖數量，前後門分開計算
            save_f_path = sub_path + "_feature_groundturth.csv"
            write_csv_data(new_feature_label, save_f_path)

            # 計算每張原圖擷取出的特徵圖數量
            count = []  # 要寫入csv，各原圖可擷取的特徵圖數量
            df = pd.read_csv(save_o_path)  # 讀取原圖的csv ，根據檔名加入特徵圖個數
            for filename in df["filename"]:
                name = Path(filename).stem
                file_carnumber = name.split("_")[0]  # 從檔名中取得 車號
                file_cardoorplace = name.split("_")[3]  # 從檔名中取得 車門位置
                file_num_door = file_carnumber + "_" + file_cardoorplace
                try:
                    count.append(feature_count[file_num_door])  # 有特徵圖
                except:
                    count.append("0")  # 沒有擷取到特徵圖
            df["count"] = count  # 儲存特徵圖的數量
            df.to_csv(save_o_path, index=None)

            print(
                "NEW original groundturth csv : ",
                save_o_path,
                " count :",
                len(new_label),
            )
            print(
                "NEW feature groundturth csv  : ",
                save_f_path,
                " count :",
                len(new_feature_label),
            )


if __name__ == "__main__":
    data_path = "merage_data"  # 原圖的資料夾路徑
    datasetpath = "dataset"  # 要儲存到的資料夾路徑
    label_path = "groundturth.csv"  # 車體的實際值csv檔
    # color = "235"                        # 選擇要納入訓練的 車色
    cartype = "all"  # 選擇要納入訓練的 車型

    # for color in ["235", "all", "NAH"]:
    for color in ["all"]:
        label = get_csv_data(label_path)  # 讀取實際值資料

        # 根據選擇的 車色，車型 選擇檔案，並創建資料集名稱
        subdataset_name, select_file_list = select_train_datatype(
            data_path, color, cartype
        )

        # 將資料集分為 train valid 並做特徵擷取， 然後建立特徵擷取圖的實際值csv檔案
        create_subdataset(select_file_list, datasetpath, subdataset_name, label)
