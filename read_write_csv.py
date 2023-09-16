import csv
import pandas as pd
def get_csv_data(csv_path):
    with open(csv_path, newline='',encoding='utf-8-sig') as csvfile:
        each_folder_cartype = []
        rows = csv.reader(csvfile)    # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
        for row in rows:            
            each_folder_cartype.append(row)
        return each_folder_cartype

def write_csv_data(data_list, save_path):
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'NID', 'S', 'L', 'R'])
        for data in data_list:
            writer.writerow(data)

def write_csv_data_test(data_list, save_path):
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([ 'Name','NID', 'S', 'L', 'R'])
        for data in data_list:
            writer.writerow(data)     

def write_csv_data_acc(data_list, save_path):
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['filename','NID', 'S', 'L', 'R','count','pred_NID','pred_S', 'pred_L', 'pred_R', 'acc_NID', 'acc_S', 'acc_L', 'acc_R'])

        for data in data_list:
            writer.writerow(data)     

def write_csv_data_acc_splitname(path):
    df = pd.read_csv(path,encoding='utf-8')
    turth_value = df.values.tolist()
    new_list = []
    for value in turth_value:
        name_list = value[0].split("_")[:4]
        print(name_list)
        new_list.append(name_list + value[1:])
        print(value)

    for value in new_list:
        print(value)
    save_path = "test.csv"
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['number','type','color','NID', 'S', 'L', 'R','count','pred_NID','pred_S', 'pred_L', 'pred_R', 'acc_NID', 'acc_S', 'acc_L', 'acc_R'])

        for data in new_list:
            writer.writerow(data)  

if __name__ == '__main__':
    # csv_path = "groundturth.csv"

    # each_folder_cartype = get_csv_data(csv_path)
    # print(each_folder_cartype)


    csv_path = "accuracy_batch16.csv"
    write_csv_data_acc_splitname(csv_path)




