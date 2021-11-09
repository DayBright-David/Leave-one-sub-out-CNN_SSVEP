# -*- coding: utf-8 -*-
"""
12-Class SSVEP EEG Dataset - Classification Using Convolutional Neural Network
User-Independent Training using Complex Spectrum Features (Leave one out)
Following implementation is an asynchronous SSVEP BCI 
using Convolutional Neural Network classification for 1 second data length
"""
import numpy as np
from numpy.core.defchararray import array
import numpy.matlib
import scipy.io as sio
# from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import numpy as np

from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.losses import categorical_crossentropy

import ssvep_utils as su

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
    'sampling_rate': 256
}

all_acc = np.zeros((10, 1))


sub_num = 10
####################
# 数据集预处理

train_data_all = np.zeros((sub_num, 720, 8, 220, 1))
labels_all = np.zeros((sub_num, 720, 12))

for subject in range(0, sub_num):

    dataset = sio.loadmat(f'data/s{subject+1}.mat')
    eeg = np.array(dataset['eeg'], dtype='float32')
    
    CNN_PARAMS['num_classes'] = eeg.shape[0]
    CNN_PARAMS['n_ch'] = eeg.shape[1]
    total_trial_len = eeg.shape[2]
    num_trials = eeg.shape[3]
    sample_rate = 256

    filtered_data = su.get_filtered_eeg(eeg, 6, 80, 4, sample_rate)
    eeg = []

    window_len = 1 
    shift_len = 1
    
    segmented_data = su.get_segmented_epochs(filtered_data, window_len, shift_len, sample_rate)
    filtered_data = []
     
    features_data = su.complex_spectrum_features(segmented_data, FFT_PARAMS)
    segmented_data = []
    
    #Combining the features into a matrix of dim [features X channels X classes X trials*segments]
    features_data = np.reshape(features_data, (features_data.shape[0], features_data.shape[1], 
                                               features_data.shape[2], features_data.shape[3]*features_data.shape[4]))
    
    train_data = features_data[:, :, 0, :].T
    #Reshaping the data into dim [classes*trials*segments X channels X features]
    for target in range(1, features_data.shape[2]):
        train_data = np.vstack([train_data, np.squeeze(features_data[:, :, target, :]).T])

    #Finally reshaping the data into dim [classes*trials*segments X channels X features X 1]    
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
    train_data_all[subject, :, :, :, :] = train_data

    total_epochs_per_class = features_data.shape[3]
    features_data = []
    
    class_labels = np.arange(CNN_PARAMS['num_classes'])
    labels = (np.matlib.repmat(class_labels, total_epochs_per_class, 1).T).ravel()
    labels = to_categorical(labels)

    labels_all[subject, :, :] = labels


########################
# Training and testing
########################
#   
    # num_folds = 10
    # kf = KFold(n_splits=num_folds, shuffle=True)
    # kf.get_n_splits(train_data)
    # cv_acc = np.zeros((num_folds, 1))
    # fold = -1

num_out = 10
loo = LeaveOneOut()
loo.get_n_splits(train_data)
cv_acc = np.zeros((num_out, 1))
out_index = -1

for train_index, test_index in loo.split(train_data_all):
    x_tr, x_ts = train_data_all[train_index], train_data_all[test_index]
    x_tr_all = np.reshape(x_tr, (x_tr.shape[0] * x_tr.shape[1], x_tr.shape[2], x_tr.shape[3], x_tr.shape[4])) 
    x_ts_all = np.reshape(x_ts, (x_ts.shape[0] * x_ts.shape[1], x_ts.shape[2], x_ts.shape[3], x_ts.shape[4]))
    
    y_tr, y_ts = labels_all[train_index], labels_all[test_index]
    y_tr_all, y_ts_all = np.reshape(y_tr, (y_tr.shape[0] * y_tr.shape[1], y_tr.shape[2])), np.reshape(y_ts, (y_ts.shape[0]*y_ts.shape[1], y_ts.shape[2]))

    input_shape = np.array([x_tr.shape[2], x_tr.shape[3], x_tr.shape[4]])
        
    # fold = fold + 1
    out_index = out_index + 1
    # 训练过程中，每一次交叉验证开始
    print("Subject:", out_index+1, "Training...")
        
    model = su.CNN_model(input_shape, CNN_PARAMS)
        
    sgd = optimizers.SGD(lr=CNN_PARAMS['learning_rate'], decay=CNN_PARAMS['lr_decay'], 
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



print(".....................................................................................")
print("Overall Accuracy Across Subjects:", np.mean(cv_acc), "%", "std:", np.std(cv_acc), "%")
print(".....................................................................................")
