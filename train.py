from config import JLNet_config
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime,date
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.base_train import base_train


class JLNet_train(base_train):
    """
    basic training class
    """

    def __init__(self, config):
        super().__init__(config)


    def objects_fun(self):
        net = self.Net.net
        # net.to(self.device)
        criterion = self.loss
        if self.optim == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.lr*0.8, momentum=self.mom, weight_decay=self.we_de)
            optimizer2 = optim.SGD(net.parameters(), lr=self.lr, momentum=self.mom, weight_decay=self.we_de)
        else:
            optimizer = optim.Adam(net.parameters(), lr=self.lr*0.8, weight_decay=self.we_de)
            optimizer2 = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.we_de)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_mil, gamma=self.lr_de)
        scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=self.lr_mil, gamma=self.lr_de)

        return net, criterion, optimizer, optimizer2,scheduler,scheduler2



    def run(self, train_id=None, test_id=None, w_num=1):
        """
        :param train_id: cross validation split list [1,2,3,4]
        :param test_id: [5]
        :param w_num: the number of validation 6-test_id
        :return:
        """

        self.writer = SummaryWriter(
            '{}/{}_kv_cross_vali0{}'.format(self.logs_name, datetime.now(), w_num))
        if train_id is None:
            train_id = [1, 2, 3, 4]
            test_id = [5]

        train_set = self.Dataset(self.list_path, self.img_root, train_id, transform=self.train_transform,
                                 sf_sequence=self.sf_sq, cross_shuffle=self.cs, sf_aln=self.sf_aln)
        test_set = self.Dataset(self.list_path, self.img_root, test_id, transform=self.test_transform, test=True)
        train_loader = DataLoader(train_set, batch_size=self.batch, shuffle=self.dl_sf)
        test_loader = DataLoader(test_set, batch_size=self.test_batch)

        fd_set = self.Dataset(self.list_path[0], self.img_root[0], test_id, transform=self.test_transform, test=True,test_each = True)
        self.fd_loader = DataLoader(fd_set, batch_size=self.test_batch)

        fs_set = self.Dataset(self.list_path[1], self.img_root[1], test_id, transform=self.test_transform, test=True,test_each = True)
        self.fs_loader = DataLoader(fs_set, batch_size=self.test_batch)

        md_set = self.Dataset(self.list_path[2], self.img_root[2], test_id, transform=self.test_transform, test=True,test_each = True)
        self.md_loader = DataLoader(md_set, batch_size=self.test_batch)

        ms_set = self.Dataset(self.list_path[3], self.img_root[3], test_id, transform=self.test_transform, test=True,test_each = True)
        self.ms_loader = DataLoader(ms_set, batch_size=self.test_batch)

        # final_loader = DataLoader(test_set, batch_size=1)
        self.train(train_loader, test_loader, w_num)
        self.writer.close()

    def train(self, train_loader, test_loader, w_num):
        ### get net,loss, optimizer
        net, criterion, optimizer, optimizer2,scheduler,scheduler2 = self.objects_fun()
        loss_fd = criterion(weight = torch.tensor([0.25,8.]).to(self.device))
        loss_fs = criterion(weight = torch.tensor([0.25,8.]).to(self.device))
        loss_md = criterion(weight = torch.tensor([0.25,8.]).to(self.device))
        loss_ms = criterion(weight = torch.tensor([0.25,8.]).to(self.device))
        loss_all= criterion(weight=torch.tensor(self.cr_weights).to(self.device))
        loss_fd_each = criterion()
        loss_fs_each = criterion()
        loss_md_each = criterion()
        loss_ms_each = criterion()
        ### add model reload
        if self.reload != '':
            checkpoints = torch.load(self.reload)
            net.load_state_dict(checkpoints['arch'])

        global_step = 0

        epoch = 0
        mmd = 0
        for epoch in range(self.epoch_num):  # loop over the dataset multiple times
            net.train()
            print('epoch: {}'.format(epoch))
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels, _, _ = data
                inputs, img_fd, img_fs, img_md, img_ms, labels, kin_fd, kin_fs, kin_md, kin_ms = data

                # inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs, img_fd, img_fs, img_md, img_ms, labels, kin_fd, kin_fs, kin_md, \
                kin_ms = inputs.to(self.device), img_fd.to(self.device), img_fs.to(self.device), \
                         img_md.to(self.device), img_ms.to(self.device), labels.to(self.device), \
                         kin_fd.to(self.device), kin_fs.to(self.device), kin_md.to(self.device), \
                         kin_ms.to(self.device)
                #
                # mmd +=1
                # if mmd == 4:
                if epoch <70:
                    ########## fd
                    optimizer.zero_grad()
                    fd = net.fd_forward(img_fd)
                    loss = loss_fd_each(fd, kin_fd)
                    loss.backward()
                    optimizer.step()
                    ########## fs
                    optimizer.zero_grad()
                    fs = net.fs_forward(img_fs)
                    loss = loss_fs_each(fs, kin_fs)
                    loss.backward()
                    optimizer.step()
                    ########## md
                    optimizer.zero_grad()
                    md = net.md_forward(img_md)
                    loss = loss_md_each(md, kin_md)
                    loss.backward()
                    optimizer.step()
                    ########## ms
                    optimizer.zero_grad()
                    ms = net.ms_forward(img_ms)
                    loss = loss_ms_each(ms, kin_ms)
                    loss.backward()
                    optimizer.step()

                    # mmd = 0
                elif epoch<160:
                    mmd +=1
                    if mmd == 4:
                        ########## fd
                        optimizer.zero_grad()
                        fd = net.fd_forward(img_fd)
                        loss = loss_fd_each(fd, kin_fd)
                        loss.backward()
                        optimizer.step()
                        ########## fs
                        optimizer.zero_grad()
                        fs = net.fs_forward(img_fs)
                        loss = loss_fs_each(fs, kin_fs)
                        loss.backward()
                        optimizer.step()
                        ########## md
                        optimizer.zero_grad()
                        md = net.md_forward(img_md)
                        loss = loss_md_each(md, kin_md)
                        loss.backward()
                        optimizer.step()
                        ########## ms
                        optimizer.zero_grad()
                        ms = net.ms_forward(img_ms)
                        loss = loss_ms_each(ms, kin_ms)
                        loss.backward()
                        optimizer.step()
                        mmd = 0

                # zero the parameter gradients
                optimizer2.zero_grad()

                # forward + backward + optimize
                fd,fs,md,ms,all = net(inputs)
                fd_label = (labels ==1).long()
                fs_label = (labels ==2).long()
                md_label = (labels ==3).long()
                ms_label = (labels ==4).long()
                ls_fd = loss_fd(fd,fd_label)
                ls_fs = loss_fs(fs,fs_label)
                ls_md = loss_md(md,md_label)
                ls_ms = loss_ms(ms,ms_label)
                ls_all = loss_all(all,labels)
                if epoch <130:
                    loss = ls_fd+ls_fs+ls_md+ls_ms+10*ls_all
                else:
                    loss = self.loss_ratio*(ls_fd+ls_fs+ls_md+ls_ms)+10*ls_all

                loss.backward()
                optimizer2.step()

                ############# not sure

                running_loss += loss.item()


                if i % self.show_lstep == (self.show_lstep - 1):
                    # print every 20 mini-batches
                    print('[epoch %d, global step %5d] loss: %.3f' %
                          (epoch + 1, global_step + 1, running_loss / self.show_lstep))

                    # ...log the running loss
                    self.writer.add_scalar(tag='training loss',
                                           scalar_value=running_loss / self.show_lstep,
                                           global_step=global_step)

                    running_loss = 0.0

                # update global step
                global_step += 1
            if self.save_tacc:
                self.acc_records(net, train_loader, epoch, t='train')
            if (epoch + 1) % self.print_frq == 0:
                self.acc_records(net, test_loader, epoch, t='test')

            # update learning rate
            scheduler.step()
            scheduler2.step()
        self.mis_record(net, test_loader, w_num, self.savemis)
        # self.acc_records(net, test_loader, epoch, t='test')
        ## save model
        if self.save_ck:
            torch.save({
                'epoch': epoch,
                'arch': net.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, '{}/{}-{}-cv{}-{}-{}.pth'.format(self.ck_pth,
                                                 self.kintype,self.model_name, w_num, date.today(),
                                                datetime.now().hour))

    def acc_records(self, net, dloader, epoch, t='train'):
        """
        :param dloader: train loader or test loader
        :param epoch: training epochs
        :param t: 'train' or 'test'
        :return: record accuracy
        """
        acc = self.Net.eval( dloader,'all')
        print('Accuracy of the network on the %s images: %d %%' % (t,100*acc))
        self.writer.add_scalar(tag='{}_accuracy/epoch'.format(t),
                               scalar_value=acc,
                               global_step=epoch)

        #####################  FD
        acc = self.Net.eval(self.fd_loader,'fd')
        print('Accuracy of the  FD network on the %s images: %d %%' % (t,100*acc))
        self.writer.add_scalar(tag='{}_fd_accuracy/epoch'.format(t),
                               scalar_value=acc,
                               global_step=epoch)

        ######################## FS
        acc = self.Net.eval(self.fs_loader,'fs')
        print('Accuracy of the  FS network on the %s images: %d %%' % (t,100*acc))
        self.writer.add_scalar(tag='{}_fs_accuracy/epoch'.format(t),
                               scalar_value=acc,
                               global_step=epoch)

        ########################## MD
        acc = self.Net.eval(self.md_loader,'md')
        print('Accuracy of the  MD network on the %s images: %d %%' % (t,100*acc))
        self.writer.add_scalar(tag='{}_md_accuracy/epoch'.format(t),
                               scalar_value=acc,
                               global_step=epoch)

        ############################# MS
        acc = self.Net.eval(self.ms_loader,'ms')
        print('Accuracy of the  MS network on the %s images: %d %%' % (t,100*acc))
        self.writer.add_scalar(tag='{}_ms_accuracy/epoch'.format(t),
                               scalar_value=acc,
                               global_step=epoch)





if __name__=='__main__':

    parser = argparse.ArgumentParser(description='train JLNet')
    parser.add_argument('--datatype','--dt',type=str,default='kfw1',help='The dataset trained on')
    args = parser.parse_args()

    if args.datatype =='kfw1':
        print('start training on kfw1')
        netmodel = JLNet_train(JLNet_config.kin_config)
        netmodel.cross_run()

    elif args.datatype == 'kfw2':
        print('start training on kfw2')
        train_ls = ['/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/fd_pairs.mat',
                    '/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/fs_pairs.mat',
                    '/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/md_pairs.mat',
                    '/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/ms_pairs.mat']
        data_pth = ['/home/wei/Documents/DATA/kinship/KinFaceW-II/images/father-dau',
                    '/home/wei/Documents/DATA/kinship/KinFaceW-II/images/father-son',
                    '/home/wei/Documents/DATA/kinship/KinFaceW-II/images/mother-dau',
                    '/home/wei/Documents/DATA/kinship/KinFaceW-II/images/mother-son']
        JLNet_config.kin_config.data_name = 'kfw2'
        JLNet_config.kin_config.epoch_num = 230
        JLNet_config.kin_config.loss_ratio = 0.3
        JLNet_config.kin_config.cr_weights = [0.2, 2, 2, 2, 2.2]
        JLNet_config.kin_config.list_path = train_ls
        ## data path
        JLNet_config.kin_config.img_root = data_pth
        netmodel = JLNet_train(JLNet_config.kin_config)

        netmodel.cross_run()

