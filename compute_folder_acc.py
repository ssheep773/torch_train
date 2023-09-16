import csv
import pandas as pd
import numpy as np
path = f"model_accuracy2.csv"
save_path = "model_accuracy_END2.csv"

all_data = []
with open(path, newline='') as csvfile:
  rows = csv.reader(csvfile)
  for row in rows:
    all_data.append(row[:8])
print(len(all_data))



best = ""
color = ''
for count in range(len(all_data)//20):
    print("-------------------------------------")
    value = {'NID':[], "S":[],"L":[],"R":[]}
    subdata = all_data[count*20:(count+1)*20]
    # print("subdata",subdata)
    for data in subdata:   # 以20行為分界 區分每個資料集的計算結果
        model_name = data[0]
        color = data[1]                         # 紀錄是什麼顏色的資料集
        best = data[2]                          # 紀錄是model 還是 best
        value[data[3]].append(data)
    
        # print(value)
    four_value = []
    for key in value.keys():   #依序計算NID S L R
        a = np.array(value[key])

        accuracy = np.average(a[:,4].astype('float'))
        acc_std = np.std(a[:,5].astype('float'), axis=0, ddof = 0)
        mae = np.average(a[:,6].astype('float'))

        accuracy = round(accuracy, 4)
        acc_std = round(acc_std, 4)
        mae = round(mae, 4)

        four_value.append([model_name, color, best, key, accuracy, acc_std, mae])
    print('four_value: ',four_value)



    
    with open(save_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['modelname', 'dataset', 'best/model', 'type', 'acc', 'std', 'mae'])
        for v in four_value:
            writer.writerow(v)   
