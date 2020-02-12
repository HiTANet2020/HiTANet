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


def train_model(training_file='training_file',
                validation_file='validation_file',
                testing_file='testing_file',
                n_diagnosis_codes=10000,
                n_labels=2,
                output_file='output_file',
                batch_size=100,
                dropout_rate=0.5,
                L2_reg=0.001,
                n_epoch=1000,
                log_eps=1e-8,
                visit_size=512,
                hidden_size=256,
                use_gpu=False,
                model_name='',
                disease = 'hf',
                code2id = None,
                running_data='',
                gamma=0.5,
                model_file = None,
                layer=1):
    options = locals().copy()

    print('building the model ...')
    model = model_file(n_diagnosis_codes, batch_size, options)
    focal_loss = FocalLoss(2, gamma=gamma)
    print('constructing the optimizer ...')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = options['L2_reg'])
    print('done!')

    print('loading data ...')
    train, validate, test = units.load_data(training_file, validation_file, testing_file)
    n_batches = int(np.ceil(float(len(train[0])) / float(batch_size)))

    print('training start')
    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    best_test_cost = 0.0
    epoch_duaration = 0.0
    best_epoch = 0.0
    max_len = 50
    best_parameters_file = ''
    if use_gpu:
        model.cuda()
    model.train()
    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)
        counter = 0

        for index in samples:
            batch_diagnosis_codes = train[0][batch_size * index: batch_size * (index + 1)]
            batch_time_step = train[2][batch_size * index: batch_size * (index + 1)]
            batch_diagnosis_codes, batch_time_step = adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes)
            batch_labels = train[1][batch_size * index: batch_size * (index + 1)]
            lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
            maxlen = np.max(lengths)
            predictions, labels, self_attention = model(batch_diagnosis_codes, batch_time_step, batch_labels, options, maxlen)
            optimizer.zero_grad()

            loss = focal_loss(predictions, labels)
            loss.backward()
            optimizer.step()

            cost_vector.append(loss.cpu().data.numpy())

            if (iteration % 50 == 0):
                print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()))
                #print(self_attention[:,0,0].squeeze().cpu().data.numpy())
                #print(time_weight[:, 0])
                #print(prior_weight[:, 0])
                #print(model.time_encoder.time_weight[0:10])
                #print(self_weight[:, 0])
            iteration += 1

        duration = time.time() - start_time
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration))

        train_cost = np.mean(cost_vector)
        validate_cost = units.calculate_cost_tran(model, validate, options, max_len, focal_loss)
        test_cost = units.calculate_cost_tran(model, test, options, max_len, focal_loss)
        print('epoch:%d, validate_cost:%f, duration:%f' % (epoch, validate_cost, duration))
        epoch_duaration += duration

        train_cost = np.mean(cost_vector)
        epoch_duaration += duration
        if validate_cost > (best_validate_cost + 0.04) and epoch > 19:
            print(validate_cost)
            print(best_validate_cost)
            break
        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_test_cost = test_cost
            best_epoch = epoch

            shutil.rmtree(output_file)
            os.mkdir(output_file)

            torch.save(model.state_dict(), output_file + model_name + '.' + str(epoch))
            best_parameters_file = output_file + model_name + '.' + str(epoch)
        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
        best_epoch, best_train_cost, best_validate_cost, best_test_cost)
        print(buf)
    # testing
    #best_parameters_file = output_file + model_name + '.' + str(8)
    model.load_state_dict(torch.load(best_parameters_file))
    model.eval()
    n_batches = int(np.ceil(float(len(test[0])) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])
    for index in range(n_batches):
        batch_diagnosis_codes = test[0][batch_size * index: batch_size * (index + 1)]
        batch_time_step = test[2][batch_size * index: batch_size * (index + 1)]
        batch_diagnosis_codes, batch_time_step = adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes)
        batch_labels = test[1][batch_size * index: batch_size * (index + 1)]
        lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
        maxlen = np.max(lengths)
        logit, labels, self_attention = model(batch_diagnosis_codes, batch_time_step, batch_labels, options, maxlen)

        if use_gpu:
            prediction = torch.max(logit, 1)[1].view((len(labels),)).data.cpu().numpy()
            labels = labels.data.cpu().numpy()
        else:
            prediction = torch.max(logit, 1)[1].view((len(labels),)).data.numpy()
            labels = labels.data.numpy()

        y_true = np.concatenate((y_true, labels))
        y_pred = np.concatenate((y_pred, prediction))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(accuary, precision, recall, f1, roc_auc)
    return (accuary, precision, recall, f1, roc_auc)

# For real data set please contact the corresponding author

if __name__ == '__main__':
    # parameters
    batch_size = 50
    dropout_rate = 0.5
    L2_reg = 1e-3
    log_eps = 1e-8
    n_epoch = 20
    n_labels = 2  # binary classification
    visit_size = 256 # size of input embedding
    hidden_size = 256 # size of hidden layer
    gamma = 0.0 # setting for Focal Loss, when it's zero, it's equal to standard cross loss
    use_gpu = True
    layer = 1 # layer of Transformer
    model_choice = 'TransformerTime' # name of the proposed HiTANet in our paper
    model_file = eval(model_choice)
    disease_list = ['hf_sample'] # name of the sample data set, you can place you own data set by following the same setting
    for disease in disease_list:
        model_name = 'tran_%s_%s_L%d_wt_1e-4_focal%.2f' % (model_choice, disease, layer, gamma)
        print(model_name)
        log_file = 'results/' + model_name + '.txt'
        path = 'data/' + disease + '/model_inputs/'
        trianing_file = path + disease + '_training_new.pickle'
        validation_file = path + disease + '_validation_new.pickle'
        testing_file = path + disease + '_testing_new.pickle'

        dict_file = 'data/' + disease + '/' + disease + '_code2idx_new.pickle'
        code2id = pickle.load(open(dict_file, 'rb'))
        n_diagnosis_codes = len(pickle.load(open(dict_file, 'rb'))) + 1

        output_file_path = 'cache/' + model_choice + '_outputs/'
        if os.path.isdir(output_file_path):
            pass
        else:
            os.mkdir(output_file_path)
        results = []
        for k in range(5):
            accuary, precision, recall, f1, roc_auc = train_model(trianing_file, validation_file,
                                                                  testing_file, n_diagnosis_codes, n_labels,
                                                                  output_file_path, batch_size, dropout_rate,
                                                                  L2_reg, n_epoch, log_eps, visit_size, hidden_size,
                                                                  use_gpu, model_name, disease=disease, code2id=code2id,
                                                                  gamma=gamma, layer=layer, model_file=model_file)
            results.append([accuary, precision, recall, f1, roc_auc])

        results = np.array(results)
        print(np.mean(results, 0))
        with open(log_file, 'w') as f:
            f.write(model_name)
            f.write('\n')
            f.write(str(np.mean(results, 0)))
            f.write('\n')
            f.write(str(np.std(results, 0)))