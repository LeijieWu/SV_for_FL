# -*- coding: utf-8 -*-
# @Time    : 2020/9/1 18:55
# @Author  : LIU YI

import numpy as np
import pandas as pd
import math
import copy

class Configs(object):

    def __init__(self):

        ## TODO For FL training
        self.data = 'noise-mnist'
        self.task_repeat_time = 500
        self.rounds = 10    #todo change update batch
        self.frac = 1
        self.user_num = 5
        self.FL_LR = 0.005
        self.model = 'cnn'
        self.iid = 0   # 0 = non-iid; 1 = iid
        self.unequal = 1   # 0 = equal; 1 = unequal
        self.gpu = 0   # 0 = CPU; 1 = GPU

        self.select = True
        self.selected_client = 0

        self.aggregation = 'avg'   # 'sv' or 'avg'


        # TODO for Fderated Env

        self.remove_client_index = None


        if self.data == 'cifar':
            # self.data_size = np.array([40, 38, 32, 46, 44]) * 250
            self.data_size = np.array([10000, 10000, 10000, 10000, 10000])
            self.batch_size = 10
            theta_num = 62006
            self.D = (self.data_size / 10) * (32 * (theta_num + 10 * (3 * 32 * 32))) / 1e9
        elif self.data == 'cifar100':
            self.data_size = np.array([40, 38, 32, 46, 44]) * 250
            self.batch_size = 10
            theta_num = 69656
            self.D = (self.data_size / 10) * (32 * (theta_num + 10 * (3 * 32 * 32))) / 1e9
        elif self.data == 'noise-mnist':
            # self.data_size = np.array([16000, 14000, 12000, 18000, 20000])
            # self.data_size = np.array([10000, 16000, 16000, 18000, 20000])
            # self.data_size = np.array([28000, 13000, 13000, 13000, 13000])
            self.data_size = np.array([ 16000, 16000, 16000, 16000, 16000])
            self.batch_size = 100
            theta_num = 21840
            self.D = (self.data_size / 10) * (32 * (theta_num + 10 * 28 * 28)) / 1e9
        else:
            # self.data_size = np.array([12000, 10000, 8000, 14000, 16000])
            # self.data_size = np.array([6000, 12000, 12000, 14000, 16000])
            # self.data_size = np.array([24000, 9000, 9000, 9000, 9000])
            self.data_size = np.array([12000, 12000, 12000, 12000, 12000])
            self.batch_size = 100
            theta_num = 21840
            self.D = (self.data_size / 10) * (32 * (theta_num + 10 * 28 * 28)) / 1e9

        self.data_size_original = copy.copy(self.data_size)



        self.frequency = np.array([1.4359949, 1.52592623, 1.04966248, 1.33532239, 1.7203678])

        if self.remove_client_index!=None:
            self.user_num = self.user_num-1
            self.data_size = np.delete(self.data_size, self.remove_client_index)
            self.frequency = np.delete(self.frequency, self.remove_client_index)
            self.D = np.delete(self.D, self.remove_client_index)


        self.C = 20
        self.alpha = 0.1
        self.local_epoch_range = 10

        self.performance = ['loss', 'acc']
        self.performance = self.performance[1]

        if self.performance == 'acc':
            if self.data == 'cifar100':
                self.lamda = 2000
            else:
                self.lamda = 1000    # todo changed for 10 rounds
        else:
            self.lamda = 4


        ## TODO For RL training

        self.EP_MAX = 2000
        self.S_DIM = self.user_num+1  # TODO add history later
        self.A_DIM = self.user_num
        self.BATCH = self.rounds  # TODO change round
        self.A_UPDATE_STEPS = 5
        self.C_UPDATE_STEPS = 5
        self.HAVE_TRAIN = False

        self.dec = 0.3
        self.A_LR = 0.00003  # todo  learning rate influence tendency
        self.C_LR = 0.00003
        self.GAMMA = 0.95
        # self.action_space = np.zeros((self.user_num, self.local_epoch_range))

        ## TODO For RL inference
        self.infer_round = 50


        ## TODO For myopia greedy

        self.myopia_frac = 1
        self.myopia_max_epoch = 1
#
#         self.lamda = 500
#
#         self.his_len = 5
#         self.info_num = 3
#
#
#
#         if self.data == 'cifar':
#             theta_num = 62006
#             data_size = np.array([40, 38, 32, 46, 44]) * 250 * 0.8
#
#         else:
#             theta_num = 21840
#             if self.user_num == 5:
#                 data_size = np.array([10000, 12000, 14000, 8000, 16000]) * 0.8
#             else:
#                 data_size = pd.read_csv('Multi_client_data/'+str(self.user_num)+'mnist.csv')
#                 data_size = np.array(data_size['data_size'].tolist())
#
#         self.D = (data_size / 10) * (32 * (theta_num + 10 * 28 * 28)) / 1e9
#         self.alpha = 0.1
#         self.tau = 1
#         self.C = 20
#         self.communication_time = np.random.uniform(low=10, high=20, size=self.user_num)
#
#         self.BATCH = 5   #todo
#         self.A_UPDATE_STEPS = 5  # origin:5
#         self.C_UPDATE_STEPS = 5
#         self.HAVE_TRAIN = False
#         self.A_LR = 0.00001  # origin:0.00003
#         self.C_LR = 0.00001
#         self.GAMMA = 0.95  # origin: 0.95
#         self.dec = 0.3
#
#         self.EP_MAX = 1000  #todo
#         self.EP_MAX_pre_train = 1000
#
#         self.EP_LEN = 100
#
#
#         if self.user_num == 5:
#             self.delta_max = np.array([1.4359949, 1.02592623, 1.54966248, 1.43532239, 1.4203678])
#
#             Loss = pd.read_csv('loss_mnist_500.csv')
#             Loss = Loss.to_dict()
#             Loss = Loss['1']
#             loss_list = []
#             for i in Loss:
#                 loss_list.append(Loss[i])
#
#
#             self.loss_list = copy.copy(loss_list)
#             num = len(loss_list)
#             buffer = 0
#             profit_increase = []
#             for i in range(0, num):
#                 loss_list[i] = -math.log(loss_list[i])
#
#             for one in loss_list:
#                 profit_increase.append(one - buffer)
#                 buffer = one
#
#             self.acc_increase_list = profit_increase
#
#
#
#         else:
#             self.delta_max = np.random.uniform(low=1, high=2, size=self.user_num)
#             data_info = pd.read_csv('Multi_client_data/' + str(self.user_num) + 'user_' + 'mnist' + '_1_0.005.csv')
#             accuracy_list = data_info['loss'].tolist()
#             num = len(accuracy_list)
#
#             self.loss_list = copy.copy(accuracy_list)
#             for i in range(0, num):
#                 accuracy_list[i] = -math.log(accuracy_list[i])
#
#             buffer = -math.log(3)
#
#             self.acc_increase_list = []
#             for one in accuracy_list:
#                 self.acc_increase_list.append(one - buffer)
#                 buffer = one
#
#         self.amplifier_baseline = np.max(self.delta_max * self.tau * self.C * self.D * self.alpha)
#         self.amplifier_hrl = np.sum(self.delta_max * self.tau * self.C * self.D * self.alpha)
#         self.reducer_baseline = 1
#         self.reducer_hrl = 100
#
#         reducer_pretrain_dict = {
#             5: 1000,
#             10: 500,
#             20: 50,
#             30: 50,
#             40: 50,
#             40: 50,
#             50: 50
#         }
#         self.reducer_pretrain = reducer_pretrain_dict[self.user_num]
#
#
if __name__ == '__main__':

    np.random.seed(2)
    configs = Configs()
    E = configs.frequency * configs.frequency * configs.C * configs.D * configs.alpha
    print(configs.user_num)
    data = 0.001 * configs.data_size
    print(E)
    print(data)