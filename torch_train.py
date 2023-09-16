import os, torch
import numpy as np
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim

# from torchinfo import summary
import pandas as pd
import tqdm
from model_accuracy import model_accuracy
import time
import copy
from datetime import datetime
from collections import OrderedDict
import math


def img_pad(pil_file, fixed_size, type1="side"):
    # print(pil_file.shape)
    try:
        h, w, c = pil_file.size
    except:
        h, w = pil_file.size
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


# import swats
# Make use of a GPU or MPS (Apple) if one is available. (see Module 3.2)
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
        # image = img_pad(image, 512)
        image = self.transform(image)  # Transform image
        label = np.array(self.labels[idx])
        return image, label  # return 模型訓練所需的資訊


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Transformer
train_transformer = transforms.Compose(
    [
        # transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

val_transformer = transforms.Compose(
    [
        # transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)


def get_DataLoader(data_dir, label_csv, value_type, transformer):
    df = pd.read_csv(label_csv)

    inputs = df["filename"].values
    labels = df[value_type].values

    for i in range(0, len(inputs)):
        inputs[i] = os.path.join(data_dir, inputs[i])

    dataloader = DataLoader(
        orangelPeelDataset(inputs, labels, transformer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    return dataloader


def trainning(dataset_path, model_name, model, batch_size=16):
    # 參數設定
    # lr = 1e-6                                        # Learning Rate
    epochs = 200  # epoch 次數

    dataset_name = os.path.split(dataset_path)[-1]
    # value_list = ["NID", "S", "L", "R"]
    value_list = ["S", "L"]
    for value_type in value_list:
        print(value_type)

        fig_dir = model_name
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        set_dolder = os.path.join(fig_dir, dataset_name)
        if not os.path.isdir(set_dolder):
            os.makedirs(set_dolder)

        train_data_dir = os.path.join(dataset_path, "train", "feature")  # 資料夾名稱
        train_label_csv = os.path.join(dataset_path, "train_feature_groundturth.csv")
        val_data_dir = os.path.join(dataset_path, "valid", "feature")  # 資料夾名稱
        val_label_csv = os.path.join(dataset_path, "valid_feature_groundturth.csv")

        train_dataloader = get_DataLoader(
            train_data_dir, train_label_csv, value_type, train_transformer
        )
        val_dataloader = get_DataLoader(
            val_data_dir, val_label_csv, value_type, val_transformer
        )

        # MODEL

        optimizer_name = "Adam"
        # optimizer_name = "SGDM"

        # optimizer = optim.Adam(model.parameters())
        # optimizer = swats.SWATS(model.parameters())
        class RMSELoss(nn.Module):
            def __init__(self, eps=1e-6):
                super().__init__()
                self.mse = nn.MSELoss()
                self.eps = eps

            def forward(self, yhat, y):
                loss = torch.sqrt(self.mse(yhat, y) + self.eps)
                if torch.isnan(loss):
                    print("loss is nan")

                return loss.float()

        # Loss function
        criterion = RMSELoss()  # 選擇想用的 loss function
        loss_name = "RMSE"
        # criterion = nn.MSELoss()

        tloss_all = []
        vloss_all = []
        best_vloss = 0
        # epoch = 0
        start = time.time()
        for epoch in range(epochs):
            start_ep = time.time()

            lr = 1e-4 if epoch < 100 else 1e-6
            # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) # 選擇你想用的 optimizer
            optimizer = optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=False
            )

            steps = list(enumerate(train_dataloader))
            pbar = tqdm.tqdm(steps)
            model.train()
            for i, (x_batch, y_batch) in pbar:
                optimizer.zero_grad()
                y_batch_pred = model(x_batch.to(device)).flatten()
                loss = criterion(y_batch_pred.float(), y_batch.to(device).float())

                loss.backward()
                optimizer.step()

                loss, current = loss.item(), (i + 1) * len(x_batch)

                if i == len(steps) - 1:
                    model.eval()
                    vloss = 0
                    count = 0
                    for i, (x_test, y_test) in enumerate(val_dataloader):
                        pred = model(x_test.to(device)).flatten()
                        # print(pred)
                        batch_vloss = criterion(pred.float(), y_test.to(device).float())
                        vloss += batch_vloss.cpu().detach()
                        count += 1

                    vloss = vloss / count
                    tloss_all.append(loss)
                    vloss_all.append(vloss)

                    if epoch == 0:
                        best_vloss = vloss
                    if best_vloss > vloss:
                        best_vloss = vloss
                        best_model_name = "best_%s_%s_EP%d_%s_batch%d_%s.pt" % (
                            model_name,
                            value_type,
                            epochs,
                            dataset_name,
                            batch_size,
                            optimizer_name,
                        )
                        BEST_PATH = os.path.join(fig_dir, dataset_name, best_model_name)
                        torch.save(model, BEST_PATH)

                    pbar.set_description(
                        f"{value_type} {dataset_name}, Epoch: {epoch}, lr:{lr}, tloss: {loss:>7f}, vloss: {vloss:>7f}, bloss:{best_vloss:>7f}"
                    )

                    txt_filename = "model_%s_%s_EP%d_%s_batch%d_%s.txt" % (
                        model_name,
                        value_type,
                        epochs,
                        dataset_name,
                        batch_size,
                        optimizer_name,
                    )
                    txt_filepath = os.path.join(fig_dir, dataset_name, txt_filename)
                    f = open(txt_filepath, "a")  # 與AA_DL同一層
                    f.write(
                        f"{value_type} Epoch: {epoch}, tloss: {loss:>7f}, vloss: {vloss:>7f}, bloss:{best_vloss:>7f}\n"
                    )
                    f.close()

                else:
                    pbar.set_description(
                        f"{value_type} {dataset_name}, Epoch: {epoch}, lr:{lr}, tloss: {loss:>7f} "
                    )
            end_ep = time.time()
            # print("time cost :", end_ep - start_ep)
        end = time.time()
        print("time cost :", end - start)
        f = open(txt_filepath, "a")  # 與AA_DL同一層

        # f.write(datetime.now())
        f.write(
            f"model:{model_name}, lr:{lr}, optimizer: {optimizer_name} ,loss: {loss_name} \n"
        )
        f.write(f"time cost : {end-start}" + "\n")
        f.write(
            "============================= dividing line =============================\n"
        )
        f.close()

        save_model_name = "model_%s_%s_EP%d_%s_batch%d_%s.pt" % (
            model_name,
            value_type,
            epochs,
            dataset_name,
            batch_size,
            optimizer_name,
        )
        PATH = os.path.join(fig_dir, dataset_name, save_model_name)
        torch.save(model, PATH)

        plt.figure()
        plt.plot(tloss_all, label="trainloss")  # plot your loss
        plt.plot(vloss_all, label="validloss")
        plt.legend()
        plt.title(
            "%s_%s_EP%d_%s_batch%d_%s"
            % (model_name, value_type, epochs, dataset_name, batch_size, optimizer_name)
        )
        plt.ylabel("loss"), plt.xlabel("epoch")
        plt.savefig(
            os.path.join(
                fig_dir,
                dataset_name,
                "model_%s_%s_EP%d_%s_batch%d_%s.png"
                % (
                    model_name,
                    value_type,
                    epochs,
                    dataset_name,
                    batch_size,
                    optimizer_name,
                ),
            )
        )
        plt.close()

        # print(y_test.cpu().detach())
        # Plot the chart
        # chart_regression(pred.flatten().cpu().detach(), y_test.cpu().detach())

    # model_accuracy(dataset_path, fig_dir, batch_size)


if __name__ == "__main__":
    batch_size = 16  # Batch Size
    model_list = []

    model = models.mobilenet_v2().to(device)
    classifier = list(model.classifier.children())  # 取得 classifier 層
    classifier[-1] = nn.Linear(1280, 1)
    model.classifier = nn.Sequential(*classifier).to(device)
    for size in [10, 20, 40]:
        model_name = f"mobilenetv2lowpass{size}"
        dataset_path = f"datasetlowpass{size}"
        train_list = [
            f"{dataset_path}/235-kfold-0-149",
            f"{dataset_path}/235-kfold-1-149",
            f"{dataset_path}/235-kfold-2-149",
            f"{dataset_path}/235-kfold-3-149",
            f"{dataset_path}/235-kfold-4-149",
            f"{dataset_path}/NAH-kfold-0-46",
            f"{dataset_path}/NAH-kfold-1-46",
            f"{dataset_path}/NAH-kfold-2-46",
            f"{dataset_path}/NAH-kfold-3-46",
            f"{dataset_path}/NAH-kfold-4-46",
            f"{dataset_path}/all_kfold-0-298",
            f"{dataset_path}/all_kfold-1-298",
            f"{dataset_path}/all_kfold-2-298",
            f"{dataset_path}/all_kfold-3-298",
            f"{dataset_path}/all_kfold-4-298",
        ]
        for dataset_path in train_list:
            print(model_name)
            trainning(dataset_path, model_name, model, batch_size)

    # model_name = "VGG16"
    # model = models.vgg16(pretrained=True).to(device)    # 使用內建的 model
    # classifier = list(model.classifier.children())      # 取得 classifier 層
    # classifier[-1] = nn.Sequential(
    #                 nn.Linear(4096, 1),
    #             ).to(device)                               # 將最後一層全連接層改為 sigmoid 函數
    # model.classifier = nn.Sequential(*classifier)
    # model_list.append((model, model_name))

    # model_name = "ResNet50"
    # model = models.resnet50(pretrained=True).to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 1).to(device)
    # model_list.append((model, model_name))

    # model_name = "densenet121"
    # model = models.densenet121(pretrained=True).to(device)
    # classifier = nn.Sequential(OrderedDict([
    #                         ('fc1', nn.Linear(1024, 1)),]))
    # model.classifier = classifier.to(device)
    # model_list.append((model, model_name))
