from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow_datasets as tfds
import scipy.interpolate as inter
import pylab as plt
import task_pb2
import sys
import argparse
from datetime import datetime
import json
import numpy as np
import operator
import os
import statistics
from scipy import interpolate
import tensorflow as tf
from sklearn.model_selection import train_test_split

class EOD_Preprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def _read_EOD_data(self):
        self.data_EOD = []
        for index, ticker in enumerate(self.tickers):
            single_EOD = np.genfromtxt(
                os.path.join(self.data_path, self.market_name + '_' + ticker +
                             '_30Y.csv'), dtype=str, delimiter=',',
                skip_header=True
            )
            self.data_EOD.append(single_EOD)

        print('#stocks\' EOD data readin:', len(self.data_EOD))
        assert len(self.tickers) == len(self.data_EOD), 'length of tickers ' \
                                                        'and stocks not match'

    def _read_tickers(self, ticker_fname):
        self.tickers = np.genfromtxt(ticker_fname, dtype=str, delimiter='\t',
                                     skip_header=True)[:, 0]

    def _transfer_EOD_str(self, selected_EOD_str, tra_date_index):
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
        for row, daily_EOD in enumerate(selected_EOD_str):
            date_str = daily_EOD[0].replace('-05:00', '')
            date_str = date_str.replace('-04:00', '')
            selected_EOD[row][0] = tra_date_index[date_str]
            '''what if data_str doesn't have one date in tra_date_index causing the index not continuous???
            so tra_date_index must have all dates for all tickers, but some ticker might not have one date in 
            tra_date_index'''
            for col in range(1, selected_EOD_str.shape[1]):
                selected_EOD[row][col] = float(daily_EOD[col])
        return selected_EOD

    '''
        Transform the original EOD data collected from Google Finance to a
        friendly format to fit machine learning model via the following steps:
            Calculate moving average (5-days, 10-days, 20-days, 30-days),
            ignoring suspension days (market open, only suspend this stock)
            Normalize features by (feature - min) / (max - min)
    '''

    def generate_feature(self, selected_tickers_fname, begin_date, opath,
                         return_days=1, pad_begin=29):
        trading_dates = np.genfromtxt(
            os.path.join(self.data_path, '..',
                         self.market_name + '_aver_line_dates.csv'), # prepare a list of dates to exclude holidays
            dtype=str, delimiter=',', skip_header=False
        )
        print('#trading dates:', len(trading_dates))
        print('begin date:', begin_date)
        # transform the trading dates into a dictionary with index
        index_tra_dates = {}
        tra_dates_index = {}
        for index, date in enumerate(trading_dates): # make a dictionary to translate dates into index and vice versa
            tra_dates_index[date] = index
            index_tra_dates[index] = date
        self.tickers = np.genfromtxt(
            os.path.join(self.data_path, '..', selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        # # for debug purpose
        # self.tickers = self.tickers[:10]
        print('#tickers selected:', len(self.tickers))
        self._read_EOD_data()
        stocks_features = []
        labels = []

        # below codes to handle relation
        industry_ticker_file = os.path.join('data/relation/sector_industry/',
                                            self.market_name + '_industry_ticker.json')
        selected_tickers = self.tickers
        print('#tickers selected:', len(selected_tickers))
        ticker_index = {}
        for index, ticker in enumerate(selected_tickers):
            ticker_index[ticker] = index  # note the key is string, content is number index
        with open(industry_ticker_file, 'r') as fin:
            industry_tickers = json.load(fin)
        print('#industries: ', len(industry_tickers))

        # concatenate lists of all tickers of all industry
        all_tickers_from_industry = sum(industry_tickers.values(), [])
        len_of_industries = len(all_tickers_from_industry)
        print('len_of_industries', len_of_industries)
        assert (set(self.tickers) == set(sum(industry_tickers.values(), [])))

        valid_industry_count = 0
        valid_industry_index = {}
        for industry in industry_tickers.keys():
            # if len(industry_tickers[industry]) > 1: # cancel this because I want every stock to have relation type
            valid_industry_index[industry] = valid_industry_count
            valid_industry_count += 1

        industry_all_tickers = {}
        for industry in valid_industry_index.keys():
            cur_ind_tickers = industry_tickers[industry]

            ind_ind = valid_industry_index[industry]
            for i in range(len(cur_ind_tickers)):
                industry_all_tickers[cur_ind_tickers[i]] = ind_ind
                print("in the industry_tickers: ", cur_ind_tickers[i])
                if cur_ind_tickers[i] in all_tickers_from_industry:
                    print("in the industry_tickers: ??", cur_ind_tickers[i])

        assert (set(self.tickers) == set(list(industry_all_tickers.keys())))
        for index, ticker in enumerate(all_tickers_from_industry):
            industry_relation = industry_all_tickers[ticker]
            # the index of self.tickers and self.data_EOD is the same. So this step to get the index in these two lists
            stock_index = ticker_index[ticker]
            single_EOD = self.data_EOD[stock_index]

            begin_date_row = -1
            for date_index, daily_EOD in enumerate(single_EOD):
                date_str = daily_EOD[0].replace('-05:00', '')
                date_str = date_str.replace('-04:00', '')
                cur_date = datetime.strptime(date_str, self.date_format)
                if cur_date > begin_date: # from original raw stock price data file extract data, if larger than our specified data than
                    begin_date_row = date_index
                    break
            selected_EOD_str = single_EOD[begin_date_row:]
            selected_EOD = self._transfer_EOD_str(selected_EOD_str, # transfer this selected date string into index
                                                  tra_dates_index)

            # calculate moving average features
            begin_date_row = -1
            for row in selected_EOD[:, 0]:
                row = int(row)
                if row >= pad_begin:  # offset for the first 30-days average
                    begin_date_row = row
                    break
            mov_aver_features = np.zeros(
                [selected_EOD.shape[0], 8], dtype=float
            )  # 4 columns refers to 5-, 10-, 20-, 30-days average

            for row in range(begin_date_row, selected_EOD.shape[0]):
                date_index = selected_EOD[row][0]
                aver_5 = 0.0
                aver_10 = 0.0
                aver_20 = 0.0
                aver_30 = 0.0
                count_5 = 0
                count_10 = 0
                count_20 = 0
                count_30 = 0
                std_5 = []
                std_10 = []
                std_20 = []
                std_30 = []
                for offset in range(30):
                    date_gap = date_index - selected_EOD[row - offset][0]
                    if date_gap < 5:
                        count_5 += 1
                        aver_5 += selected_EOD[row - offset][4]
                        std_5.append(selected_EOD[row - offset][4])
                    if date_gap < 10:
                        count_10 += 1
                        aver_10 += selected_EOD[row - offset][4]
                        std_10.append(selected_EOD[row - offset][4])
                    if date_gap < 20:
                        count_20 += 1
                        aver_20 += selected_EOD[row - offset][4]
                        std_20.append(selected_EOD[row - offset][4])
                    if date_gap < 30:
                        count_30 += 1
                        aver_30 += selected_EOD[row - offset][4]
                        std_30.append(selected_EOD[row - offset][4])
                if count_5 == 0: # some data such as ticker DWAQ in period of 2016-12-06~21 missing data 15 days
                    mov_aver_features[row][0] = -1234
                else:
                    mov_aver_features[row][0] = aver_5 / count_5
                if count_10 == 0:
                    mov_aver_features[row][1] = -1234
                else:
                    mov_aver_features[row][1] = aver_10 / count_10
                mov_aver_features[row][2] = aver_20 / count_20
                mov_aver_features[row][3] = aver_30 / count_30
                if len(std_5) <= 1:
                    mov_aver_features[row][4] = -1234
                else:
                    mov_aver_features[row][4] = statistics.stdev(std_5)
                if len(std_10) <= 1:
                    mov_aver_features[row][5] = -1234
                else:
                    mov_aver_features[row][5] = statistics.stdev(std_10)
                mov_aver_features[row][6] = statistics.stdev(std_20)
                mov_aver_features[row][7] = statistics.stdev(std_30)

            print('industry_relation:', industry_relation)
            '''
                normalize features by feature / max, the max price is the
                max of close prices, I give up to subtract min for easier
                return ratio calculation.
            '''
            pri_min = np.min(selected_EOD[begin_date_row:, 4])
            price_max = np.max(selected_EOD[begin_date_row:, 4])

            print(self.tickers[stock_index], 'minimum:', pri_min,
                  'maximum:', price_max, 'ratio:', price_max / pri_min)
            if price_max / pri_min > 10:
                print('!!!!!!!!!')

            '''
                generate feature and ground truth in the following format:
                date_index, 5-day, 10-day, 20-day, 30-day, close price
                two ways to pad missing dates:
                for dates without record, pad a row [date_index, -1234 * 5]

                delete the part with /price_max for close price
                deal missing value with spline since in automl-zero masking and deleting weights later in loss function
                do not work
            '''
            features = np.ones([len(trading_dates) - pad_begin, 16],
                               dtype=float) * -1234
            rows = np.ones([len(trading_dates) - pad_begin, 1],
                           dtype=float)
            # data missed at the beginning
            for row in range(len(trading_dates) - pad_begin):
                rows[row][0] = row
            for row in range(begin_date_row, selected_EOD.shape[0]):
                cur_index = int(selected_EOD[row][0])
                features[cur_index - pad_begin][0:8] = mov_aver_features[
                    row]

                '''adding the next if condition because of the above mentioned prob - index might not be continuous. 
                    Only if continuous will add features'''
                if cur_index - int(selected_EOD[row - return_days][0]) == \
                        return_days:
                    features[cur_index - pad_begin][-8:-3] = \
                        selected_EOD[row][1:]
                    if (row + return_days) < selected_EOD.shape[0]:
                        if selected_EOD[row + return_days][-2] > selected_EOD[row][-2]:
                            features[cur_index - pad_begin][-2] = index * 10 + 1
                            if selected_EOD[row][-2] == 0: # selected_EOD[row + return_days][-2] == 0 or comment off because unlike log return raw return can be 0
                                features[cur_index - pad_begin][-1] = -1234
                            else:
                                features[cur_index - pad_begin][-1] = (selected_EOD[row + return_days][-2] - selected_EOD[row][-2]) / selected_EOD[row][-2]
                            if np.abs((selected_EOD[row + return_days][-2] - selected_EOD[row][-2])/selected_EOD[row][-2]) > 1:
                                print('np.abs(selected_EOD[row + return_days][-2] - selected_EOD[row][-2])!!!!!!!!!!!!!!', np.abs(selected_EOD[row + return_days][-2] - selected_EOD[row][-2]))
                                print()
                                sys.exit()
                        else:
                            features[cur_index - pad_begin][-2] = index * 10
                            if selected_EOD[row][-2] == 0: # selected_EOD[row + return_days][-2] == 0 or comment off because unlike log return raw return can be 0
                                features[cur_index - pad_begin][-1] = -1234
                            else:
                                features[cur_index - pad_begin][-1] = (selected_EOD[row + return_days][-2] - selected_EOD[row][-2]) / selected_EOD[row][-2]
                            if np.abs((selected_EOD[row + return_days][-2] - selected_EOD[row][-2])/selected_EOD[row][-2]) > 1:
                                print('np.abs(selected_EOD[row + return_days][-2] - selected_EOD[row][-2])!!!!!!!!!!!!!!', np.abs(selected_EOD[row + return_days][-2] - selected_EOD[row][-2]))
                                sys.exit()

            '''volume number is big so log before normalize'''
            features[:, -4][features[:, -4] > 0] = np.log(features[:, -4][features[:, -4] > 0])
            '''generating pairs would require this column to have pair number'''
            ''' rhe reason why below codes is to allow binary task generation. Then setting label to -1234 to
            make sure that this row is not used in evaluation'''
            features[:, -2][features[:, -2] == -1234] = index * 10
            features[:, -1][features[:, -2] == -1234] = -1234
            if np.sum(list(features[:, -2] == -1234)) != 0:
                print('how many number are assign positive: ', np.sum(list(features[:, -2] == -1234)))
                print('out of these numbers how many are features have')
            if (features[:, :-2] == 0).any(): # for the stock DWPP whose 0 is missing values
                features[:, :-2][features[:, :-2] == 0] = -1234
                print('np.sum(list(features == 5))', np.sum(list(features == -1234)))

            features[:, -3] = industry_relation
            features = np.delete(features, -1, 0) # last row all missing as show in feng's github
            features = np.delete(features, -1, 0) # because the last row don't have label since no tomorrow data
            labels.append(features[:, -2:])
            for j in range(len(features[:, -1])):
                if np.abs(features[:, -1][j]) > 1 and np.abs(features[:, -1][j]) != 1234:
                    print('> 1!!!!!!!!!!', features[:, -1][j])
                    print('index', j)
                    print('features[j-1]', features[j-1])
                    print('features[j]', features[j])
                    print('features[j+1]', features[j+1])
                    print(self.tickers[stock_index])
                    sys.exit()

            features = np.delete(features, 14, 1)
            features = np.delete(features, 14, 1)
            max_num = np.max(features[:, :4][features[:, :4] != -1234])
            min_num = np.min(features[:, :4][features[:, :4] != -1234])
            max_num2 = np.max(features[:, 8:12][features[:, 8:12] != -1234])
            min_num2 = np.min(features[:, 8:12][features[:, 8:12] != -1234])
            max_vol = np.max(features[:, 4:8][features[:, 4:8] != -1234])
            min_vol = np.min(features[:, 4:8][features[:, 4:8] != -1234])
            max_volume = np.max(features[:, 12][features[:, 12] != -1234])
            min_volume = np.min(features[:, 12][features[:, 12] != -1234])
            max_num = np.maximum(max_num2, max_num)
            min_num = np.minimum(min_num2, min_num)
            for i in range(np.shape(features)[1]):
                not_mask_column = features[:, i][features[:, i] != -1234]
                if i in [0, 1, 2, 3] or i in [8, 9, 10, 11]:
                    features[:, i][features[:, i] != -1234] = (not_mask_column - min_num)/(max_num - min_num)
                elif i in [4, 5, 6, 7]:
                    features[:, i][features[:, i] != -1234] = (not_mask_column - min_vol)/(max_vol - min_vol)
                elif i in [12]:
                    features[:, i][features[:, i] != -1234] = (not_mask_column - min_volume) / (max_volume - min_volume)
            if np.shape(features)[0] != 1244:
                sys.exit()

            assert (sum(list(features[:, 13] == -1234)) == 0)
            assert (sum(list(features[:, -1] == -1234)) == 0)
            stocks_features.append(features)
        for i in range(len(labels)):
            for j in range(len(labels[i][:, 1])):
                if np.abs(labels[i][:, 1][j]) > 1 and np.abs(labels[i][:, 1][j]) != 1234:
                    print('> 1!!!!!!!!!!', labels[i][:, 1][j])
                    print('index', j)
                    print(self.tickers[i])
                    sys.exit()
        labels = np.vstack(labels) # not vstack because the vector is no longer vertical
        stocks_features = np.vstack(stocks_features)
        np.save('stocks_features.npy', stocks_features)
        np.save('labels.npy', labels)
        X_train, X_test, y_train, y_test = train_test_split(stocks_features, labels, test_size=0.33, random_state=42, shuffle=False)
        return X_train, X_test, y_train, y_test

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

flags.DEFINE_string(
    'data_dir', 'binary_cifar10_data_labels_as_price_diff_normalized_by_each_stock_no_shuffle_at_all_relation_save_npy',
    'Path of the folder to save the datasets.')

flags.DEFINE_string(
    'tfds_data_dir', '/tmp/',
    'Path for tensorflow_datasets to cache downloaded datasets, '
    'only used in local runs.')

flags.DEFINE_integer('num_train_examples', 1000,
                     'Number of training examples in each dataset.')

flags.DEFINE_integer('num_valid_examples', 244,
                     'Number of validation examples in each dataset.')

flags.DEFINE_integer('num_test_examples', 0,
                     'Number of test examples in each dataset.')
''' I change test example to 0 because in run_baseline.py there is no test examples and final evaluate tasks use 
different tasks than search and select. So final evaluate no need test examples'''
flags.DEFINE_integer('projected_dim', 13,
                     'The dimensionality to project the data into.')

flags.DEFINE_string('dataset_name', 'cifar10',
                    'Name of the dataset to generatee '
                    'more binary classification datasets.')

flags.DEFINE_integer('min_data_seed', 0,
                     'Generate one dataset for each seed in '
                     '[min_data_seed, max_data_seed).')

flags.DEFINE_integer('max_data_seed', 1,
                     'Generate one dataset for each seed in '
                     '[min_data_seed, max_data_seed).')

flags.DEFINE_list('class_ids', None,
                  'Classes included to generate binary'
                  ' classification datasets.')

FLAGS = flags.FLAGS


def create_projected_binary_dataset(
    dataset_name, positive_class, negative_class,
    num_train_examples, num_valid_examples, num_test_examples,
    projected_dim, seed, load_fn):
  """Create a projected binary dataset from the given spec and seed."""
  num_samples = (
      num_train_examples +
      num_valid_examples +
      num_test_examples)
  pos = positive_class
  neg = negative_class
  data, labels, _, _ = get_dataset(
      dataset_name,
      None, [pos, neg], load_fn=load_fn) # didn't use the second argument to limit the sample size of pos/neg class
  labels = labels[np.in1d(labels[:, 0], [pos, neg])] # from first column axis0 find pos and neg index and return the labels
  labels = np.delete(labels, 0, 1) # delete one dimension because the first dimension is the number I set to get pairs, e.g. 801 and 800

  (train_data, train_labels, valid_data, valid_labels,
   test_data, test_labels) = train_valid_test_split(
       data, labels,
       num_train_examples,
       num_valid_examples,
       num_test_examples,
       seed)

  dataset = task_pb2.ScalarLabelDataset()
  for i in range(train_data.shape[0]):
    train_feature = dataset.train_features.add()
    train_feature.features.extend(list(train_data[i]))
    dataset.train_labels.append(train_labels[i])
    if np.abs(train_labels[i]) > 1 and np.abs(train_labels[i]) != 1234:
        print('train_labels[i]!!!!!!!!!!!!!!',
              train_labels[i])
        sys.exit()
  for i in range(valid_data.shape[0]):
    valid_feature = dataset.valid_features.add()
    valid_feature.features.extend(list(valid_data[i]))
    dataset.valid_labels.append(valid_labels[i])
    if np.abs(valid_labels[i]) > 1 and np.abs(valid_labels[i]) != 1234:
        print('valid_labels[i]!!!!!!!!!!!!!!',
              valid_labels[i])
        sys.exit()
  if test_data is not None:
    for i in range(test_data.shape[0]):
      test_feature = dataset.test_features.add()
      test_feature.features.extend(list(test_data[i]))
      dataset.test_labels.append(test_labels[i])
  return dataset


def load_projected_binary_dataset(saved_dataset):
  """Load the binary dataset saved in a ScalarLabelDataset proto."""
  num_train = len(saved_dataset.train_labels)
  assert len(saved_dataset.train_labels) == len(saved_dataset.train_features)
  num_valid = len(saved_dataset.valid_labels)
  assert len(saved_dataset.valid_labels) == len(saved_dataset.valid_features)
  num_test = len(saved_dataset.test_labels)
  assert len(saved_dataset.test_labels) == len(saved_dataset.test_features)
  if num_train == 0 or num_valid == 0:
    raise ValueError('Number of train/valid examples'
                     ' must be more than zero.')
  feature_size = len(saved_dataset.train_features[0].features)

  train_data = np.zeros((num_train, feature_size))
  train_labels = np.zeros(num_train)
  for i in range(num_train):
    train_labels[i] = saved_dataset.train_labels[i]
    for j in range(feature_size):
      train_data[i][j] = saved_dataset.train_features[i].features[j]

  valid_data = np.zeros((num_valid, feature_size))
  valid_labels = np.zeros(num_valid)
  for i in range(num_valid):
    valid_labels[i] = saved_dataset.valid_labels[i]
    for j in range(feature_size):
      valid_data[i][j] = saved_dataset.valid_features[i].features[j]

  if num_test > 0:
    test_data = np.zeros((num_test, feature_size))
    test_labels = np.zeros(num_test)
    for i in range(num_test):
      test_labels[i] = saved_dataset.test_labels[i]
      for j in range(feature_size):
        test_data[i][j] = saved_dataset.test_features[i].features[j]
  else:
    test_data = None
    test_labels = None

  return (train_data, train_labels, valid_data, valid_labels,
          test_data, test_labels)


def get_dataset(
    name, num_samples_per_class=None, class_ids=None, load_fn=None,
    data_dir=None):

  # Load datasets.
  dataset_dict = load_fn(
      name, data_dir=data_dir, batch_size=-1)
  # Whether the dataset is from tfds or given in unit test.
  if load_fn == tfds.load:
    train_data = dataset_dict['X_train']
    train_labels = dataset_dict['y_train']
    test_data = dataset_dict['X_test']
    test_labels = dataset_dict['y_test']
  else:
    train_data = dataset_dict['X_train']
    train_labels = dataset_dict['y_train']
    test_data = dataset_dict['X_test']
    test_labels = dataset_dict['y_test']

  train_data = train_data.astype(np.float)
  test_data = test_data.astype(np.float)
  assert train_data.shape[0] == train_labels.shape[0]
  assert test_data.shape[0] == test_labels.shape[0]

  if class_ids is not None:
    train_data = train_data[np.where(np.isin(train_labels[:, 0], class_ids))]
    train_labels = train_labels[np.where(np.isin(train_labels[:, 0], class_ids))]
    test_data = test_data[np.where(np.isin(test_labels[:, 0], class_ids))]
    test_labels = test_labels[np.where(np.isin(test_labels[:, 0], class_ids))]
  assert train_data.shape[0] == train_labels.shape[0]
  assert test_data.shape[0] == test_labels.shape[0]

  train_data = np.concatenate((train_data, test_data), axis=0)
  train_labels = np.concatenate((train_labels, test_labels), axis=0)
  return (train_data, train_labels, test_data, test_labels)

def train_valid_test_split(
    data, labels,
    num_train_examples, num_valid_examples, num_test_examples,
    seed, use_stratify=False):
  """Split data into train, valid and test with given seed."""
  ''' change stratefy=False in the context of changing lalebs into values and coding financial measures'''
  if num_test_examples > 0:
    if use_stratify:
      stratify = labels
    else:
      stratify = None
    train_data, test_data, train_labels, test_labels = (
        sklearn.model_selection.train_test_split(
            data, labels,
            train_size=(
                num_train_examples +
                num_valid_examples),
            test_size=num_test_examples,
            random_state=seed, shuffle=False, stratify=stratify))

  else:
    train_data, train_labels = data, labels
    test_data = None
    test_labels = None
  if use_stratify:
    stratify = train_labels
  else:
    stratify = None
  # where shuffle happens
  train_data, valid_data, train_labels, valid_labels = (
      sklearn.model_selection.train_test_split(
          train_data, train_labels,
          train_size=num_train_examples,
          test_size=num_valid_examples,
          random_state=seed, shuffle=False, stratify=stratify))
  return (
      train_data, train_labels,
      valid_data, valid_labels,
      test_data, test_labels)


def main(unused_argv):
  """Create and save the datasets."""
  del unused_argv
  desc = "pre-process EOD data market by market, including listing all " \
         "trading days, all satisfied stocks (5 years & high price), " \
         "normalizing and compansating data"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-path', help='path of EOD data')
  parser.add_argument('-market', help='market name')
  args = parser.parse_args()

  if args.path is None:
      args.path = 'data/google_finance'
  if args.market is None:
      args.market = 'NASDAQ'

  processor = EOD_Preprocessor(args.path, args.market)

  X_train, X_test, y_train, y_test = processor.generate_feature(
      processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv',
      datetime.strptime('2012-11-19 00:00:00', processor.date_format),
      os.path.join(processor.data_path, '..', '2013-01-01'), return_days=1,
      pad_begin=29
  )
  FLAGS.class_ids = list(range(0, 10*len(processor.tickers), 10))+list(range(1, 10*len(processor.tickers), 10))
  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  tfds_cached_dict = {}
  data_dir = FLAGS.tfds_data_dir if FLAGS.tfds_data_dir else None
  name = FLAGS.dataset_name
  dataset_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
  tfds_cached_dict[name] = dataset_dict

  # To mock the API of tfds.load to cache the downloaded datasets.
  # Used as an argument to `get_dataset`.
  def load_fn(name, data_dir=None, batch_size=-1):
    # This function will always return the whole dataset.
    assert batch_size == -1
    del data_dir
    del batch_size
    return tfds_cached_dict[name]
  class_ids = sorted([int(x) for x in FLAGS.class_ids])
  for i in class_ids:
    for j in class_ids:
        if j == (i-1):
          print('Generating pos {} neg {}'.format(i, j))
          positive_class = i
          negative_class = j
          random_seeds = range(FLAGS.min_data_seed, FLAGS.max_data_seed)
          for seed in random_seeds:
            dataset = create_projected_binary_dataset(
                FLAGS.dataset_name, positive_class, negative_class,
                FLAGS.num_train_examples, FLAGS.num_valid_examples,
                FLAGS.num_test_examples, FLAGS.projected_dim, seed, load_fn)
            filename = 'binary_{}-pos_{}-neg_{}-dim_{}-seed_{}'.format(
                FLAGS.dataset_name, positive_class, negative_class,
                FLAGS.projected_dim, seed)
            serialized_dataset = dataset.SerializeToString()

            with open(os.path.join(FLAGS.data_dir, filename), 'wb') as f:
              f.write(serialized_dataset)

if __name__ == '__main__':
  app.run(main)
