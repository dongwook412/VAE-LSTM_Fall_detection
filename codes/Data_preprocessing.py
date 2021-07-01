# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

train_path = 'C:/Users/thehb/OneDrive/Desktop/성훈_석사/수업/고급시계열/프로젝트/VAE-LSTM-for-anomaly-detection/datasets/NAB-known-anomaly/csv-files/fall_data/train'
# test_path = 'C:/Users/thehb/OneDrive/Desktop/성훈_석사/수업/고급시계열/프로젝트/VAE-LSTM-for-anomaly-detection/datasets/NAB-known-anomaly/csv-files/fall_data/val'
test_path = 'C:/Users/thehb/OneDrive/Desktop/성훈_석사/수업/고급시계열/프로젝트/VAE-LSTM-for-anomaly-detection/datasets/NAB-known-anomaly/csv-files/fall_data/test'

file_list = []
train_file_list = os.listdir(train_path)
test_file_list = os.listdir(test_path)
for file in train_file_list:
    file_list.append(os.path.abspath(train_path+'/'+file))
for file in test_file_list:
    file_list.append(os.path.abspath(test_path+'/'+file))


all_data = []
all_label = []
window_size = 10 # 0.05초

for file_dir in file_list: # train dataset 크기(size) 구할 때 file_list[0:train파일갯수]
    print(file_dir)
    df = pd.read_csv(file_dir, usecols=['gyro_z', 'label'])   
    df.loc[(df.label == 'FOL'), 'label'] = 1
    df.loc[(df.label == 'FKL'), 'label'] = 1
    df.loc[(df.label == 'BSC'), 'label'] = 1
    df.loc[(df.label == 'SDL'), 'label'] = 1
    df.loc[(df.label != 1), 'label'] = 0
    data = np.array(df.gyro_z)
    label = np.array(df.label)
    for i in range(len(data) // window_size):        
        all_data.append(np.mean(data[i*window_size:i*window_size+window_size]))
        all_label.append(0 if np.sum(label[i*window_size:i*window_size+window_size]) < 5 else 1)

len(all_data)

#pd.DataFrame(all_data, columns=['values']).to_csv('C:/Users/thehb/OneDrive/Desktop/성훈_석사/수업/고급시계열/프로젝트/VAE-LSTM-for-anomaly-detection/datasets/NAB-known-anomaly/csv-files/fall_gyro_z.csv', header= True, index=True)
#pd.DataFrame(all_label, columns=['label']).to_csv('C:/Users/thehb/OneDrive/Desktop/성훈_석사/수업/고급시계열/프로젝트/VAE-LSTM-for-anomaly-detection/datasets/NAB-known-anomaly/csv-files/fall_label.csv', header= True, index=True)
pd.DataFrame(all_data, columns=['values']).to_csv('C:/Users/thehb/OneDrive/Desktop/성훈_석사/수업/고급시계열/프로젝트/VAE-LSTM-for-anomaly-detection/datasets/NAB-known-anomaly/csv-files/test_fall_gyro_z.csv', header= True, index=True)
pd.DataFrame(all_label, columns=['label']).to_csv('C:/Users/thehb/OneDrive/Desktop/성훈_석사/수업/고급시계열/프로젝트/VAE-LSTM-for-anomaly-detection/datasets/NAB-known-anomaly/csv-files/test_fall_label.csv', header= True, index=True)


############################
# train dataset 크기(size) 구할 때
size = 0
for file_dir in file_list[0:13]: # train dataset 크기(size) 구할 때 file_list[0:train파일갯수]
    print(file_dir)
    df = pd.read_csv(file_dir, usecols=['gyro_z', 'label'])   
    size += len(df)
    # if size > 350000:
    #     print('zzzzzzzzzzz')
    #     print(file_dir)
    #     break
print(size)
###############################

