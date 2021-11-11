#! user/bin/env python3
# -*- coding: utf-8 -*-c

'''
    Project:
        预训练模型
    Author:
        Bo Dai
    version:
        2021.11.09

'''
import numpy as np
from numpy.core.records import array
import numpy.matlib
from scipy.io.matlab.mio import loadmat
import scipy.io as sio
import ssvep_utils as su
from sklearn.model_selection import LeaveOneOut

from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.losses import categorical_crossentropy

import os
import sys
current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(target_dir)
from SSVEP_BCI_Eval.baselines import cca, fbcca, trca, cnn


folder = 'SSVEP_BCI_Eval/datasets/Benchmark_35sub'

fs = 250

ch_names=[ 'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
        'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
        'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M1',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
        'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
        'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4',
        'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']
# 导联选择
chs = [47,53, 54,55,56,57,60,61,62]
[ch_names[c] for c in chs]
len_gaze_s = 5
len_shift_s = 0.5
len_delay_s = 0.13
len_sel_s = len_gaze_s + len_shift_s
# Data length [samples]
len_gaze_sampl = np.around(len_gaze_s * fs)
# Visual latency [samples]
len_delay_smpl = np.around(len_delay_s)

selection_time = len_gaze_s + len_shift_s

frequencies=[8., 9., 10., 11., 12., 13., 14., 15., 8.2, 9.2,
            10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 8.4, 9.4, 10.4, 11.4,
            12.4, 13.4, 14.4, 15.4, 8.6, 9.6, 10.6, 11.6, 12.6, 13.6,
              14.6, 15.6, 8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8]
502947
phase=[0., 1.57079633, 3.14159265, 4.71238898, 0.,
      1.57079633, 3.14159265, 4.71238898, 1.57079633, 3.14159265,
      4.71238898, 0., 1.57079633, 3.14159265, 4.71238898,
      0., 3.14159265, 4.71238898, 0., 1.57079633,
      3.14159265, 4.71238898, 0., 1.57079633, 4.71238898,
      0., 1.57079633, 3.14159265, 4.71238898, 0.,
      1.57079633, 3.14159265, 0., 1.57079633, 3.14159265,
      4.71238898, 0., 1.57079633, 3.14159265, 4.71238898]


CNN_PARAMS = {
    'batch_size': 256,
    'epochs': 50,
    'droprate': 0.25,
    'learning_rate': 0.001,
    'lr_decay': 0.0,
    'l2_lambda': 0.005,
    'momentum': 0.9,
    'kernel_f': 10,
    'n_ch': 8,
    'num_classes': 12}

FFT_PARAMS = {
    'resolution': 0.2930,
    'start_frequency': 3.0,
    'end_frequency': 35.0,
    'sampling_rate': 250
}

sub_num = 35
# (10, 35, 1)
all_acc = np.zeros((int(np.floor_divide(len_gaze_s, 0.5)), sub_num, 1))

# 加载数据cc
####################
# 数据集预处理

train_data_all = np.zeros((sub_num, 40*6*5, 9, 220, 1))    # (10, 35, 40*6, 8, 220, 1), 基于时间窗，我需要再加一维，
labels_all = np.zeros((sub_num, 40*6*5, 40))



for subject in range(0, sub_num):

    print(f'For S{subject+1}: ')
    data = loadmat(f'{folder}/S{subject+1}.mat')
    eeg = data['data'].transpose((1, 0, 2, 3))    # 1500, 64, 40, 6
    # eeg = eeg[int(((0.5 + 0.136) * fs)):int(((0.5 + 0.136) * fs + (1.0 * fs))), chs, :, :]    # 125, 9, 40, 6
    eeg = eeg[:, chs, :, :]
    samples, channels, targets, blocks = eeg.shape
    # ()
    CNN_PARAMS['num_classes'] = eeg.shape[2]
    CNN_PARAMS['n_ch'] = eeg.shape[1]
    total_trial_len = eeg.shape[0]
    num_trials = eeg.shape[3]
    sample_rate = 250
    # (1250, 9, 40, 6) / 应该是(40, 9, 1250, 6)
    filtered_data = su.get_filtered_eeg(eeg, 6, 80, 4, sample_rate)
    eeg = []

    window_len = 1 
    shift_len = 1
    # (40, 9, 6, 5, 250)
    segmented_data = su.get_segmented_epochs(filtered_data.transpose((2, 1, 0, 3)), window_len, shift_len, sample_rate)
    filtered_data = []
    # (220, 9, 40, 6, 5)
    features_data = su.complex_spectrum_features(segmented_data, FFT_PARAMS)
    segmented_data = []
    # (220, 9, 40, 30)
    #Combining the features into a matrix of dim [features X channels X classes X trials*segments]
    features_data = np.reshape(features_data, (features_data.shape[0], features_data.shape[1], 
                                               features_data.shape[2], features_data.shape[3]*features_data.shape[4]))
    # (30, 9, 220)
    train_data = features_data[:, :, 0, :].T
    #Reshaping the data into dim [classes*trials*segments X channels X features]
    for target in range(1, features_data.shape[2]):
        train_data = np.vstack([train_data, np.squeeze(features_data[:, :, target, :]).T])
    # (1200, 9, 220)
    #Finally reshaping the data into dim [classes*trials*segments X channels X features X 1]    
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
    train_data_all[subject, :, :, :, :] = train_data
    # (35, 1200, 9, 220, 1)
    total_epochs_per_class = features_data.shape[3]
    features_data = []
    
    class_labels = np.arange(CNN_PARAMS['num_classes'])
    labels = (np.matlib.repmat(class_labels, total_epochs_per_class, 1).T).ravel()
    labels = to_categorical(labels)
    # (35, 1200, 40)
    labels_all[subject, :, :] = labels

sio.savemat('SSVEP_BCI_Eval/baselines/data_for_training/1s_segmented_filtered_data_for_35subs.mat', {'array': train_data_all})
# (35, 1200, 9, 220, 1)
sio.savemat('SSVEP_BCI_Eval/baselines/data_for_training/1s_labels_for_35subs.mat', {'array': labels_all})
# (35, 1200, 40)



# 数据预处理完毕：
# 先滤波（5s整体），再切分，得到40*6*5个样本
# train_data_all: (35, 1200, 9, 220, 1)
# labels_all: (35, 1200, 40)


#####################################################################
# Training and testing

num_out = sub_num
loo = LeaveOneOut()
loo.get_n_splits(train_data)
cv_acc = np.zeros((num_out, 1))
out_index = -1
# 输出一个(35, 10)维的准确度矩阵
acc_cnn_complex_spectrum_for_35sub = np.zeros((sub_num, int(np.floor_divide(len_gaze_s, 0.5))))

for train_index, test_index in loo.split(train_data_all):
    x_tr, x_ts = train_data_all[train_index], train_data_all[test_index]
    x_tr_all = np.reshape(x_tr, (x_tr.shape[0] * x_tr.shape[1], x_tr.shape[2], x_tr.shape[3], x_tr.shape[4])) 
    x_ts_all = np.reshape(x_ts, (x_ts.shape[0] * x_ts.shape[1], x_ts.shape[2], x_ts.shape[3], x_ts.shape[4]))
    # (34, 1200, 40) (1, 1200, 40)
    y_tr, y_ts = labels_all[train_index], labels_all[test_index]
    y_tr_all, y_ts_all = np.reshape(y_tr, (y_tr.shape[0] * y_tr.shape[1], y_tr.shape[2])), np.reshape(y_ts, (y_ts.shape[0]*y_ts.shape[1], y_ts.shape[2]))
    # (40800, 40) (1200, 40)
    input_shape = np.array([x_tr.shape[2], x_tr.shape[3], x_tr.shape[4]])
    # (9, 220, 1)
    # fold = fold + 1ccc
    out_index = out_index + 1
    # 训练过程中，每一次交叉验证开始
    print("Subject:", out_index+1, "Training...")
        
    model = cnn.CNN_model(input_shape, CNN_PARAMS)
        
    sgd = optimizers.gradient_descent_v2.SGD(lr=CNN_PARAMS['learning_rate'], decay=CNN_PARAMS['lr_decay'], 
                             momentum=CNN_PARAMS['momentum'], nesterov=False)
    model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=["accuracy"])
    history = model.fit(x_tr_all, y_tr_all, batch_size=CNN_PARAMS['batch_size'], 
                        epochs=CNN_PARAMS['epochs'], verbose=0)

    score = model.evaluate(x_ts_all, y_ts_all, verbose=0) 
    cv_acc[out_index, :] = score[1]*100
    # 输出每一次交叉验证的结果
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    

    # all_acc[out_index] = np.mean(cv_acc)
    print("...................................................")
    print("Subject:", out_index+1, " - Accuracy:", cv_acc[out_index, :],"%")
    print("...................................................")










