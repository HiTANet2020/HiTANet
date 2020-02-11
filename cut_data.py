#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:27:59 2019

@author: ffm5105
"""

import os
import pickle
import time
import random
import shutil
import models.units as units
from models.units import FocalLoss, adjust_input
from models.transformer import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

if __name__ == '__main__':
    # parameters
    batch_size = 50
    dropout_rate = 0.5
    L2_reg = 1e-3
    log_eps = 1e-8
    n_epoch = 20
    n_labels = 2  # binary classification
    visit_size = 256
    hidden_size = 256
    gamma = 0.0 # setting for Focal Loss, when it's zero, it's equal to standard cross loss
    use_gpu = True
    layer = 1 # layer of Transformer
    model_choice = 'TransformerTime'
    model_file = eval(model_choice)
    disease_list = ['hf']
    for disease in disease_list:
        model_name = 'tran_%s_%s_L%d_wt_1e-4_focal%.2f' % (model_choice, disease, layer, gamma)
        print(model_name)
        log_file = 'results/' + model_name + '.txt'
        path = 'data/' + disease + '/model_inputs/'
        training_file = path + disease + '_training_new.pickle'
        validation_file = path + disease + '_validation_new.pickle'
        testing_file = path + disease + '_testing_new.pickle'
        train, validate, test = units.cut_data(training_file, validation_file, testing_file)
        path_new = 'data/' + disease + '_sample' + '/model_inputs/'
        training_file_sample = path_new + disease + '_sample'  + '_training_new.pickle'
        validation_file_sample = path_new + disease + '_sample'  + '_validation_new.pickle'
        testing_file_sample = path_new + disease + '_sample' + '_testing_new.pickle'
        pickle.dump(train, open(training_file_sample, 'wb'))
        pickle.dump(validate, open(validation_file_sample, 'wb'))
        pickle.dump(test, open(testing_file_sample, 'wb'))