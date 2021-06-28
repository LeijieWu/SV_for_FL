import os
import copy
import time
import json
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt        #matplotlib inline
import csv

import torch
from tensorboardX import SummaryWriter
from RL_brain import PPO
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LeNet
from utils import get_dataset, average_weights, exp_details, sv_weights
import pandas as pd
import random
import threading
import itertools

from configs import Configs
from DNC_PPO import PPO
from itertools import product

def sv_top80percent_data(file_name):
    with open(file_name, 'r') as load_f:
        load_sv_dict = json.load(load_f)

    print("The lenth of read idx group {} is {}.".format(file_name, len(load_sv_dict.keys())))

    data_idx = []
    data_sv = []
    for idx in list(load_sv_dict.keys()):  # Extract the information in the dict
        data_idx.append(float(idx))
        # if len(load_sv_dict[idx]) < 220:
        #     print('Lenth of {} sv list is {}'.format(idx, len(load_sv_dict[idx])))
        load_sv_dict[idx] = sum(load_sv_dict[idx])/len(load_sv_dict[idx])
        data_sv.append(float(load_sv_dict[idx]))
    # print(len(data_idx), len(data_sv))


    # sort idx according to sv
    idxs_sv = np.vstack((data_idx, data_sv))
    idxs_sv = idxs_sv[:, idxs_sv[1, :].argsort()]
    idxs_sorted = idxs_sv[0, :]    # idxs with sv from small to large
    # print(idxs_sorted)
    idxs_reverse_sorted = idxs_sorted[::-1]  # idxs with sv from large to small
    # print(idxs_reverse_sorted)
    top_80percent_sv_data_idxs = idxs_reverse_sorted[:int(0.6 * len(idxs_reverse_sorted))]   # select idxs with top 80% sv
    # print(top_80percent_sv_data_idxs, len(top_80percent_sv_data_idxs))
    return top_80percent_sv_data_idxs

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class Env(object):

    def __init__(self, configs):
        self.configs = configs
        self.data_size = configs.data_size
        self.frequency = configs.frequency
        self.C = configs.C
        self.lamda = configs.lamda
        self.seed = 0
        self.D = configs.D
        self.history_avg_price = np.zeros(self.configs.user_num)
        # self.lamda = 0.8
        self.file_name = ["../src/task500_normal_noniid_client0_v1.json",
                          "../src/task500_normal_noniid_client1_v1.json",
                          "../src/task500_normal_noniid_client2_v1.json",
                          "../src/task500_normal_noniid_client3_v1.json",
                          "../src/task500_normal_noniid_client4_v1.json"]

    def reset(self):
        self.index = 0
        self.data_value = 0.001 * self.data_size
        self.unit_E = self.configs.frequency * self.configs.frequency * self.configs.C * self.configs.D * self.configs.alpha  #TODO
        self.bid = self.data_value + self.unit_E
        self.bid_ = np.zeros(self.configs.user_num)
        self.action_history = []
        self.user_groups_all = []

        # todo annotate these random seed if run greedy, save them when run DRL
        # np.random.seed(self.seed)
        # torch.random.manual_seed(self.seed)
        # random.seed(self.seed)
        # torch.cuda.manual_seed_all(self.seed)
        # torch.cuda.manual_seed(self.seed)

        start_time = time.time()
        self.acc_list = []
        self.loss_list = []
        # define paths
        path_project = os.path.abspath('..')
        self.logger = SummaryWriter('../logs')

        self.args = args_parser()
        exp_details(self.args)

        if self.configs.gpu:
            # torch.cuda.set_device(self.args.gpu)
            # device = 'cuda' if args.gpu else 'cpu'

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = 'cpu'

        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)
        print("11111111:", self.user_groups)

        # read_dictionary = np.load('user_groups_normal_non_iid_1.npy', allow_pickle=True).item()
        # print("22222222:", read_dictionary)
        # self.user_groups = copy.deepcopy(read_dictionary)
        # print("33333333:", self.user_groups)

        # TODO select data with top sv.
        # for i in range(self.configs.user_num):
        #     self.user_groups[i] = sv_top80percent_data(self.file_name[i])
        #     print("The selected top idx group of client {} is {}, and its lenth is {}.\n".format(i, self.user_groups[i], len(self.user_groups[i])))

        # TODO  random select data.
        # for i in range(self.configs.user_num):
        #     self.user_groups[i] = np.random.choice(read_dictionary[i], int(0.6 * len(read_dictionary[i])), replace=False)
        #     print(self.user_groups[i], len(self.user_groups[i]), read_dictionary[i], len(read_dictionary[i]))


        # count = 0
        # for i in range(self.configs.user_num):
        #     for idx in self.user_groups[i]:
        #         if idx not in read_dictionary[i]:
        #             count += 1
        # if count != 0:
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # else:
        #     print("#################################")



        if self.configs.remove_client_index != None:
            self.user_groups.pop(self.configs.remove_client_index)

        # BUILD MODEL
        if self.args.model == 'cnn':
            # Convolutional neural netork
            if self.args.dataset == 'mnist':
                self.global_model = CNNMnist(args=self.args)
            elif self.args.dataset == 'fmnist':
                self.global_model = CNNFashion_Mnist(args=self.args)
            elif self.args.dataset == 'cifar':
                self.global_model = CNNCifar(args=self.args)
            elif self.args.dataset == 'cifar100':
                self.global_model = LeNet()

        elif self.args.model == 'mlp':
            # Multi-layer preceptron
            img_size = self.train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                self.global_model = MLP(dim_in=len_in, dim_hidden=64,
                                   dim_out=self.args.num_classes)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.

        print(get_parameter_number(self.global_model))
        print('---------------------------------------------------------------------------------------')

        self.global_model.to(device)
        self.global_model.train()
        print(self.global_model)

        # copy weights
        global_weights = self.global_model.state_dict()

        # Training
        self.train_loss, self.train_accuracy = [], []
        self.test_loss, self.test_accuracy = [], []
        self.acc_before = 0
        self.loss_before = 300
        self.val_acc_list, self.net_list = [], []
        self.cv_loss, self.cv_acc = [], []
        self.print_every = 1
        val_loss_pre, counter = 0, 0


    def step(self, action, round):

        self.local_weights, self.local_losses = [], []
        print(f'\n | Global Training Round : {self.index + 1} |\n')

        pass
        self.global_model.train()
        idxs_users = np.array(list(self.user_groups.keys()))

        # TODO  DRL Action

        action = 5 * action
        action = action.astype(int)

        #TODO FedAvg here
        # tep = 3
        # action = np.array([tep, tep, tep, tep, tep])

        self.action_history = list(self.action_history)
        self.action_history.append(action)
        self.action_history = np.array(self.action_history)
        # print("Action", action)
        # print(type(action))
        self.local_ep_list = action

        # all_idx_dict_client_0 = {i: np.array([]) for i in self.user_groups[0]}   # sv dict for all idx of client 0


        #TODO single thread
        for idx in idxs_users:

            local_ep = self.local_ep_list[list(idxs_users).index(idx)]

            if idx == 0:
                print("########## The Lenth of {} is {}".format(self.user_groups[idx], len(self.user_groups[idx])))


            if local_ep != 0:
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                          idxs=self.user_groups[idx], logger=self.logger)

                w, loss = local_model.update_weights(
                    model=copy.deepcopy(self.global_model), global_round=self.index, local_ep=local_ep)
                self.local_weights.append(copy.deepcopy(w))
                self.local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(self.local_weights)
        self.global_model.load_state_dict(global_weights)
        test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)
        print('Avg Aggregation Test Accuracy: {:.2f}% \n'.format(100 * test_acc))

        self.index += 1


def Hand_control():
    configs = Configs()
    env = Env(configs)
    # random.seed(env.seed)

    all_idx_sv_dict = {}
    # recording = pd.DataFrame([], columns=['state history', 'action history', 'reward history', 'acc increase hisotry', 'time hisotry', 'energy history', 'social welfare', 'accuracy', 'time', 'energy'])

    # for i in range(configs.task_repeat_time):
    #     print("####### This is the {} repeat task ########".format(i))
    env.reset()

    for t in range(configs.rounds):
        local_ep_list = [1, 1, 1, 1, 1]
        action = np.array(local_ep_list)/5
        print(action)
        env.step(action, t)


    # recording = recording.append([{'state history': state_list, 'action history': action_list, 'reward history':reward_list, 'acc increase hisotry': performance_increase_list, 'time hisotry': time_list, 'energy history': energy_list, 'social welfare': np.sum(reward_list), 'accuracy': np.sum(performance_increase_list), 'time': np.sum(time_list), 'energy': np.sum(energy_list)}])
    # recording.to_csv('Hand_control_result.csv')



if __name__ == '__main__':
    # sv_top80percent_data(file_name[0])
    Hand_control()