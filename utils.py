#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from numpy.linalg import norm
from numpy import linalg as LA
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_noniid_unequal


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                #raise NotImplementedError()
                user_groups = cifar_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def Cosine_Sim_weights(w,g):

    #normalization parameter
    n = 0

    #calculate weights
    w_avg = w[0].copy() #use to extract dict name

    comparison_list = []

    all_global_weight = np.array([])
    all_client_weight = np.array([])

    for key in w_avg.keys():
        global_weight = torch.flatten(g[key],start_dim=0)
        global_weight = global_weight.numpy()
        all_global_weight = np.append(all_global_weight,global_weight)

    for i in range(len(w)):
        all_client_weight = np.array([])
        for key in w_avg.keys():
            client_weight = torch.flatten(w[i][key],start_dim=0)
            client_weight = client_weight.numpy()
            all_client_weight = np.append(all_client_weight,client_weight)
        cos_sim = (np.dot(all_client_weight,all_global_weight))/(norm(all_global_weight)*norm(all_client_weight))

        comparison_list.append(cos_sim)

    comparison_list = np.array(comparison_list)

    comparison_list += n #normalization parameter

    cos_sim_sum = sum(comparison_list)
    comparison_list = comparison_list / cos_sim_sum #update each clients' weights

    #update global
    for key in w_avg.keys():
        g[key] = g[key].zero_()

    for i in range(len(w)):
        for key in w_avg.keys():
            g[key] += w[i][key] * comparison_list[i]

    return g

def FedProx(w,g):

    mu = 0.1
    w_avg = w[0].copy() #use to extract dict name

    for key in w_avg.keys():
        local_temp = (w[0][key] - g[key]) #difference between g & l
        #local_norm = local_norm.numpy()
        local_norm = LA.norm(local_temp)
        local_prox_term = np.square(local_norm)*mu*0.5
        #print(key,w[0][key])
        #print("Prox",local_prox_term)
        w[0][key] += local_prox_term


    #create all new local update
    for key in w_avg.keys():
        for i in range(1, len(w)):
            local_temp = (w[i][key] - g[key])
            #local_norm = local_norm.numpy()
            local_norm = LA.norm(local_temp)
            local_prox_term = np.square(local_norm)*mu*0.5
            w[i][key] += local_prox_term
            w[0][key] += w[i][key]
            #print(i,"-----------------",w[0][key])
        w[0][key] = torch.div(w[0][key],len(w))

    return w[0]

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys(): #keys to obtain name of weight/bias
        for i in range(1, len(w)): #len w same as no of client, start with 1
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
