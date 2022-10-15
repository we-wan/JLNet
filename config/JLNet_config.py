import torch.nn as nn
import utils.loader
from models import modules
from utils.transform import *
from utils.loader import *
# from utils.loss import *


####################### KinfaceW I&II ############################
class kin_config(object):
    ################### data ####################

    des = 'JLNet'
    ################### data ####################
    ## kinship typle e.g. F-D, M-D, F-S etc.
    data_name = 'kfw1'
    kintype  = 'all'
    ## list path
    list_path = ['/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/fd_pairs.mat',
                '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/fs_pairs.mat',
                '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/md_pairs.mat',
                '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/ms_pairs.mat']
    ## data path
    img_root  = ['/home/wei/Documents/DATA/kinship/KinFaceW-I/images/father-dau',
                '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/father-son',
                '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/mother-dau',
                '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/mother-son']
    ## dataset
    Dataset =  KinDataset_condufusion2
    ## transformer
    trans =  [train_transform,test_transform]
    ################### loader ####################

    sf_sequence = True
    ## shuflle the list after each epoch
    cross_shuffle = True
    ##
    sf_aln = True
    ###
    dataloder_shuffle = True

    ################### train ####################
    ######## model

    ## modelname
    model = modules
    ## structure
    model_name = 'JLNet'
    ## image size
    imsize = (6,64,64)
    ## epoch numbers
    epoch_num = 200
    epoch_stone1 = 70
    epoch_stone2 = 160
    epoch_stone3 = 130
    ## batch
    train_batch =  64
    test_batch = 64
    ## learning rate
    lr    = 0.0001
    ## learning rate decay
    lr_decay = 0.5
    ##
    momentum = 0.9

    ## loss weights
    loss_ratio = 0.4
    loss_all_ratio=10

    ## criterion weights
    cr_weights = [0.18,2,2,2,2.2]
    ##
    # lr_milestones = [180, 250, 300, 400, 500, 550]
    lr_milestones = [80,150]
    ## regularization
    weight_decay = 5e-3

    ## number of cross validation
    cross_num = 5
    ## how many steps show the loss
    show_lstep = 4
    ##  frequent of printing the evaluation acc
    prt_fr = 100
    ## loss
    loss = nn.CrossEntropyLoss
    ## optimal
    optimal = 'adam'

    ######## record
    ## save the training accuracy
    save_tacc = False
    ## whether load pretrained model
    reload = ''
    ## save trained model
    save_ck = True
    ##
    logs_name = 'data/logs'
    ##
    savemis = False
    ##
    save_graph = False
