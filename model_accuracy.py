import os, torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
import pandas as pd
import tqdm
from read_write_csv import write_csv_data_test, write_csv_data_acc
import time
import copy
import csv
from datetime import datetime
from collections import OrderedDict

np.seterr(divide="ignore", invalid="ignore")

# 參數設定
# batch_size = 16
device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
# print(f"Using device: {device}")


class orangelPeelDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames  # 資料集的所有檔名
        self.labels = labels  # 影像的標籤
        self.transform = transform  # 影像的轉換方式

    def __len__(self):
        return len(self.filenames)  # return DataSet 長度

    def __getitem__(self, idx):  # idx: Inedx of filenames
        image = Image.open(self.filenames[idx]).convert("RGB")
        image = self.transform(image)  # Transform image
        label = np.array(self.labels[idx])
        return image, label  # return 模型訓練所需的資訊


# Transformer
val_transformer = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop(256),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_DataLoader(data_dir, label_csv, value_type, transformer, batch_size):
    df = pd.read_csv(label_csv)
    inputs = df["filename"].values
    labels = df[value_type].values

    for i in range(0, len(inputs)):
        inputs[i] = os.path.join(data_dir, inputs[i])

    dataloader = DataLoader(
        orangelPeelDataset(inputs, labels, transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return dataloader


def model_accuracy(dataset_path, model_folder, batch_size):
    dataset_name = os.path.split(dataset_path)[-1]  # 從dataset_path 中取出資料集名稱
    print("目前計算準確的資料集: ", dataset_name)

    model_list = []
    value_list = ["NID", "S", "L", "R"]
    time_list = []

    result = {"NID": [], "S": [], "L": [], "R": []}
    for value_type in value_list:  # 依序預測4個橘皮數值
        val_data_dir = os.path.join(dataset_path, "test", "feature")  # 資料夾名稱
        val_label_csv = os.path.join(dataset_path, "test_feature_groundturth.csv")
        val_dataloader = get_DataLoader(
            val_data_dir, val_label_csv, value_type, val_transformer, batch_size
        )

        # Load

        for weight in os.listdir(f"{model_folder}/{dataset_name}"):
            # print(weight)
            if (
                weight[-2:] == "pt" and value_type == weight.split("_")[2]
            ):  # 篩選出權重黨 # 選擇目前要預測的橘皮數值類型的model 像是選NID的模型
                if weight[:5] == "model":  # 選擇 best 的模型
                    PATH = os.path.join(f"{model_folder}/{dataset_name}", weight)
        print(f"抓取 {value_type} weight檔案位置: ", PATH)
        model_list.append(PATH)
        model = torch.load(PATH)  # 載入模型
        model.eval()  # 進入評估狀態
        start = time.time()
        for i, (x_test, y_test) in enumerate(val_dataloader):
            pred = model(x_test.to(device)).flatten().cpu().detach()
            pred = pred.numpy()
            result[value_type] = result[value_type] + list(pred)  # 把所以有同類數值存在一個list中
        end = time.time()
        time_list.append(end - start)

    write_list = []
    for value_type in value_list:
        write_list.append(result[value_type])  # 將每4個一維同類數值，合在一起

    write_list = np.array(write_list)
    write_list = write_list.T  # 將儲存的數值矩陣轉置 變成儲存的格式

    df = pd.read_csv(val_label_csv)

    name = df.values.tolist()

    name_with_pred = np.concatenate([name, write_list], axis=1)
    turth_value = name_with_pred.tolist()

    acc_list = []  # 計算整體的準確度
    for i in range(len(turth_value)):  # 計算每張圖的準確度
        turth = np.array(turth_value[i][1:5]).astype("float")
        pred = np.array(turth_value[i][5:10]).astype("float")
        acc = 1 - abs((pred - turth) / turth)
        turth_value[i] = turth_value[i] + acc.tolist()
        if not np.isnan(turth_value[i][-4:][0]):  # 排除nan 不納入整體準確度計算
            acc_list.append(turth_value[i][-4:])

    save_path = os.path.join(
        dataset_path, f"{model_folder}_feature_pred_{batch_size}.csv"
    )
    write_csv_data_test(turth_value, save_path)

    # 新增計算每張原圖的預測值，以及計算準確度
    vaild_o_path = f"{dataset_path}/test_original_groundturth.csv"
    pred_feature_list = name_with_pred

    df = pd.read_csv(vaild_o_path)  # 讀取原圖的實際值 並在後面新加入4個項目，用於儲存預測值
    turth_value = df.values.tolist()

    for i in range(len(turth_value)):
        turth_value[i] = turth_value[i] + [0, 0, 0, 0]

    # original_pred =
    for i in range(len(turth_value)):
        original_name = Path(turth_value[i][0]).stem
        count = 0
        for test in pred_feature_list:
            pred_name = test[0].split("image")[0][:-1]
            if pred_name == original_name:
                pred_value = np.array(test[5:], dtype=np.float32)  # 抓取特徵圖的預測值
                turth_value[i][-4:] = turth_value[i][-4:] + pred_value  # 根據原圖，將特徵圖數值加總
                count += 1
        turth_value[i][-4:] = np.array(turth_value[i][-4:]) / count

    acc_list = []  # 計算整體的準確度
    mae_list = []
    for i in range(len(turth_value)):  # 計算每張圖的準確度
        turth = np.array(turth_value[i][1:5]).astype("float")
        pred = np.array(turth_value[i][6:10]).astype("float")
        acc = 1 - abs((pred - turth) / turth)
        mae = abs(pred - turth)
        turth_value[i] = turth_value[i] + acc.tolist() + mae.tolist()
        if not np.isnan(turth_value[i][-8:][0]):  # 排除nan 不納入整體準確度計算
            acc_list.append(turth_value[i][-8:])

    write_csv_data_acc(
        turth_value,
        f"{dataset_path}/{model_folder}_test_accuracy_batch{batch_size}.csv",
    )
    acc_list = np.array(acc_list)
    mean_acc = np.mean(acc_list, axis=0).tolist()  # 計算準確度的平均 與 標準差
    std_acc = np.std(acc_list, axis=0, ddof=0).tolist()

    print("準確度         :", mean_acc[:4])
    print("準確度-std     :", std_acc[:4])
    print("MAE           :", mean_acc[-4:])
    print("MAE-std       :", std_acc[-4:])
    print("TIME COST (s) :", time_list)
    print("count", len(turth_value))

    write_model_acc_list = np.array(
        [model_list, mean_acc[:4], std_acc[:4], mean_acc[-4:], std_acc[-4:], time_list]
    )
    write_model_acc_list = write_model_acc_list.T.tolist()

    write_model_acc_list[0] = (
        write_model_acc_list[0] + [str(count)] + [str(datetime.now())]
    )

    for i in range(len(write_model_acc_list)):
        data = write_model_acc_list[i]
        d = data[0].split("\\")
        setname = d[0].split("/")[1]
        name_list = d[1].split("_")
        state = name_list[0]
        modelname = name_list[1]
        valurname = name_list[2]
        write_model_acc_list[i] = [
            modelname,
            setname,
            state,
            valurname,
        ] + write_model_acc_list[i][1:]

    save_path = "model_accuracy2.csv"
    with open(save_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        for data in write_model_acc_list:
            writer.writerow(data)


if __name__ == "__main__":
    batch_size = 8  # Batch Size

    # train_list = [
    #     "datasetbilater/235-kfold-0-149",
    #     "datasetbilater/235-kfold-1-149",
    #     "datasetbilater/235-kfold-2-149",
    #     "datasetbilater/235-kfold-3-149",
    #     "datasetbilater/235-kfold-4-149",
    #     "datasetbilater/NAH-kfold-0-46",
    #     "datasetbilater/NAH-kfold-1-46",
    #     "datasetbilater/NAH-kfold-2-46",
    #     "datasetbilater/NAH-kfold-3-46",
    #     "datasetbilater/NAH-kfold-4-46",
    #     "datasetbilater/all_kfold-0-298",
    #     "datasetbilater/all_kfold-1-298",
    #     "datasetbilater/all_kfold-2-298",
    #     "datasetbilater/all_kfold-3-298",
    #     "datasetbilater/all_kfold-4-298",
    # ]

    dataset_path = "datasetlowpass"
    train_list = [
        f"{dataset_path}/235-kfold-0-149",
        f"{dataset_path}/235-kfold-1-149",
        f"{dataset_path}/235-kfold-2-149",
        f"{dataset_path}/235-kfold-3-149",
        f"{dataset_path}/235-kfold-4-149",
        # f"{dataset_path}/NAH-kfold-0-46",
        # f"{dataset_path}/NAH-kfold-1-46",
        # f"{dataset_path}/NAH-kfold-2-46",
        # f"{dataset_path}/NAH-kfold-3-46",
        # f"{dataset_path}/NAH-kfold-4-46",
        # f"{dataset_path}/all_kfold-0-298",
        # f"{dataset_path}/all_kfold-1-298",
        # f"{dataset_path}/all_kfold-2-298",
        # f"{dataset_path}/all_kfold-3-298",
        # f"{dataset_path}/all_kfold-4-298",
    ]
    for dataset_path in train_list:
        model_folder = "mobilenetv2lowpass"

        model_accuracy(dataset_path, model_folder, batch_size)
