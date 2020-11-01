import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from utils.loader import *
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,fbeta_score
# from models import joint_cnn
from utils.transform import test_transform
from models.modules import JLNet
from functools import partial


class test(object):
    def __init__(self, modNet, dloader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dloader = dloader
        self.Net = modNet()


    def ki_test(self, ckpth, list_path, img_root, test_id, test_batch, ntype='', real_sn=False,test_each=False):
        """
        kinship identification test
        :return:
        """

        self.Net.load(ckpth)
        self.infer = partial(self.Net.inference,net_type=ntype)
        test_set = self.dloader(list_path, img_root, test_id, transform=test_transform, test=True, test_each=test_each,real_sn = real_sn)
        test_loader = DataLoader(test_set, batch_size=test_batch)
        total_pred = []
        total_label=[]
        self.Net.net.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels, _, _ = data
                images, labels = images.to(self.device), labels.to(self.device)
                if ntype =='cascade':
                    predicted = self.infer(images)
                else:
                    outputs = self.infer(images)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.cpu().data.numpy()
                labels = labels.cpu().data.numpy()
                total_pred = np.concatenate((total_pred, predicted), axis=0)
                total_label = np.concatenate((total_label, labels), axis=0)

        if real_sn:
            confu_m = confusion_matrix(total_label, total_pred, labels=[1, 2, 3, 4], normalize='true')
            f10_fd = fbeta_score(total_label, total_pred, labels=[1], beta=10, average='macro')
            f10_fs = fbeta_score(total_label, total_pred, labels=[2], beta=10, average='macro')
            f10_md = fbeta_score(total_label, total_pred, labels=[3], beta=10, average='macro')
            f10_ms = fbeta_score(total_label, total_pred, labels=[4], beta=10, average='macro')
            micro_f1 = fbeta_score(total_label, total_pred, beta=10, average='macro')
            acc = sum(total_label == total_pred) / len(total_label)
            return confu_m, f10_fd, f10_fs, f10_md, f10_ms, micro_f1, acc
        else:
            if test_each:
                confu_m = confusion_matrix(total_label, total_pred, labels=[1, 2, 3, 4], normalize='true')
                micro_f1 = f1_score(total_label, total_pred)
                acc = sum(total_label == total_pred) / len(total_label)
                return confu_m, micro_f1, acc
            else:
                confu_m = confusion_matrix(total_label, total_pred, labels=[1, 2, 3, 4], normalize='true')
                micro_f1 = f1_score(total_label, total_pred, average='macro')
                acc = sum(total_label == total_pred) / len(total_label)
                return confu_m, micro_f1, acc


    def cv_ki(self,ckp_pth,train_ls,data_pth,savename,test_batch,n_type='',real_sn = False,test_each=False):
        """
        cross validation kinship identification
        :param ckp_pth:
        :param train_ls:
        :param data_pth:
        :param savename:
        :return:
        """

        divnum = len(os.listdir(ckp_pth))
        if real_sn:
            con_avg = 0
            f10 = 0
            f10_fd_all = 0
            f10_fs_all = 0
            f10_md_all = 0
            f10_ms_all = 0
            avg_acc = 0
            for i, ld in enumerate(sorted(os.listdir(ckp_pth))):
                ld = os.path.join(ckp_pth, ld)
                confu_norm, f10_fd, f10_fs, f10_md, f10_ms, micro_f1, acc =self.ki_test(ckpth=ld,list_path=train_ls,
                                                         img_root=data_pth,test_id=[5-i],
                                                         test_batch=test_batch, ntype=n_type,
                                                        real_sn= real_sn,test_each=test_each)
                con_avg = confu_norm + con_avg
                f10_fd_all += f10_fd
                f10_fs_all += f10_fs
                f10_md_all += f10_md
                f10_ms_all += f10_ms
                f10 += micro_f1
                avg_acc += acc

            con_avg = con_avg / 5
            f10 = f10 / 5
            f10_fd_all /= 5
            f10_fs_all /= 5
            f10_md_all /= 5
            f10_ms_all /= 5
            avg_acc = avg_acc / 5
            f10_4avg = (f10_fd_all + f10_fs_all + f10_md_all + f10_ms_all) / 4

            print(con_avg)

            print('f10_fd:{}'.format(f10_fd_all))
            print('f10_fs:{}'.format(f10_fs_all))
            print('f10_md:{}'.format(f10_md_all))
            print('f10_ms:{}'.format(f10_ms_all))
            print('avg:{}'.format(f10_4avg))
            print('average_macro_f1:{}'.format(f10))
            print('avg acc:{:04}'.format(avg_acc))

            plt.figure()
            df_cm = pd.DataFrame(con_avg, ['F-D', 'F-S', 'M-D', 'M-S'], ['F-D', 'F-S', 'M-D', 'M-S'])
            sn.set(font_scale=0.8)  # for label size
            sn.heatmap(df_cm, vmin=0, vmax=1, cmap='Blues', annot=True, annot_kws={"size": 16})  # font size
            # plt.show()
            plt.savefig('{}-kfw-{}-combine-hm.png'.format(savename, test_data))
        else:
            con_avg = 0
            mf1 = 0
            avg_acc = 0
            for i, ld in enumerate(sorted(os.listdir(ckp_pth))):
                ld = os.path.join(ckp_pth, ld)
                confu_norm, micro_f1, acc = self.ki_test(ckpth=ld,list_path=train_ls,
                                                         img_root=data_pth,test_id=[5-i],
                                                         test_batch=test_batch, ntype=n_type,real_sn= real_sn,test_each=test_each)
                con_avg = confu_norm + con_avg
                mf1 += micro_f1
                avg_acc += acc

            con_avg = con_avg / divnum
            mf1 = mf1 / divnum
            avg_acc = avg_acc / divnum
            print(con_avg)
            print('average_macro_f1:{}'.format(mf1))
            print('average multiclass acc:{:04}'.format(avg_acc))
            plt.figure()
            # df_cm = pd.DataFrame(con_avg, ['No-kin', 'F-D', 'F-S', 'M-D', 'M-S'], ['No-kin', 'F-D', 'F-S', 'M-D', 'M-S'])
            df_cm = pd.DataFrame(con_avg, ['F-D', 'F-S', 'M-D', 'M-S'], ['F-D', 'F-S', 'M-D', 'M-S'])

            sn.set(font_scale=0.8)  # for label size
            sn.heatmap(df_cm, vmin=0, vmax=1, cmap='Blues', annot=True, annot_kws={"size": 16})  # font size
            # plt.show()
            # plt.savefig('stage3-{}_test1{}_hm{}.png'.format(number,stage3_joint_config.kin_config.model_name, '_avg'))
            plt.savefig('{}-kfw-{}-hm.png'.format(savename,test_data))



if __name__ == '__main__':

    test_data = 'I'

    train_ls = ['/home/wei/Documents/DATA/kinship/KinFaceW-{}/meta_data/fd_pairs.mat'.format(test_data),
                '/home/wei/Documents/DATA/kinship/KinFaceW-{}/meta_data/fs_pairs.mat'.format(test_data),
                '/home/wei/Documents/DATA/kinship/KinFaceW-{}/meta_data/md_pairs.mat'.format(test_data),
                '/home/wei/Documents/DATA/kinship/KinFaceW-{}/meta_data/ms_pairs.mat'.format(test_data)]
    data_pth = ['/home/wei/Documents/DATA/kinship/KinFaceW-{}/images/father-dau'.format(test_data),
                '/home/wei/Documents/DATA/kinship/KinFaceW-{}/images/father-son'.format(test_data),
                '/home/wei/Documents/DATA/kinship/KinFaceW-{}/images/mother-dau'.format(test_data),
                '/home/wei/Documents/DATA/kinship/KinFaceW-{}/images/mother-son'.format(test_data)]

    # ckp_1 = '/home/wei/Documents/CODE/kinship/data/checkpoints/kfw1_stage3_13/stage3-join_atten7_fix'
    # ckp_2 = '/home/wei/Documents/CODE/kinship/data/checkpoints-kfw2/kfw2_stage3_5/stage3-join_atten7_fix'
    ckp_1 = '/home/wei/Documents/CODE/ECCV/eccv/data/checkpoints/final/pin1-join_atten7_fix'
    ckp_2 = '/home/wei/Documents/CODE/ECCV/eccv/data/checkpoints/final/ww2-join_atten7_fix'

    ckp_dict = {'I': ckp_1, 'II': ckp_2}
    ckp_pth = ckp_dict[test_data]

    # kinship verification
    testmode = test(JLNet, KinDataset_condufusion2)
    print('test fd')
    testmode.cv_ki(ckp_pth, train_ls[0], data_pth[0], 'fd', test_batch=100, n_type='fd', real_sn=False, test_each = True)
    print('test fs')
    testmode.cv_ki(ckp_pth, train_ls[1], data_pth[1], 'fs', test_batch=100, n_type='fs', real_sn=False, test_each=True)
    print('test md')
    testmode.cv_ki(ckp_pth, train_ls[2], data_pth[2], 'md', test_batch=100, n_type='md', real_sn=False, test_each=True)
    print('test ms')
    testmode.cv_ki(ckp_pth, train_ls[3], data_pth[3], 'ms', test_batch=100, n_type='ms', real_sn=False, test_each=True)



    # kinship identification
    testmode = test(JLNet,KinDataset_condufusion2)
    testmode.cv_ki(ckp_pth,train_ls,data_pth,'try',test_batch=1000,n_type='cascade',real_sn=False)


