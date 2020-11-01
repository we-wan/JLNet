from .basic_nets import JLNet_basic
import torch


class JLNet(object):
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = JLNet_basic().to(self.device)

    def load(self,ck_pth):
        checkpoints = torch.load(ck_pth)
        self.net.load_state_dict(checkpoints['arch'])


    def inference(self,images,net_type='all'):
        if net_type == 'all':
            fd,fs,md,ms,outputs = self.net(images)
            return outputs
        elif net_type == 'fd':
            outputs = self.net.fd_forward(images)
            return outputs
        elif net_type == 'fs':
            outputs = self.net.fs_forward(images)
            return outputs
        elif net_type == 'md':
            outputs = self.net.md_forward(images)
            return outputs
        elif net_type == 'ms':
            outputs = self.net.ms_forward(images)
            return outputs
        elif net_type == 'cascade':
            pred = self.cascade(images)
            return pred


    def cascade(self,img,th1 = 0.6,th2 = 0.5):
        """
        combine multi outputs and binary outputs
        :param img:
        :param th1:
        :param th2:
        :return:
        """
        _, _, _, _, outputs = self.net(img)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().data.numpy()

        thresh1 = th1
        thresh2 = th2
        final_p = predicted
        for i, item in enumerate(predicted):
            if item == 1:
                fd_out = torch.nn.functional.softmax(self.net.fd_forward(img[i:i + 1]), dim=1)
                if fd_out.data[:, 1].item() < thresh1:
                    fd_out = torch.nn.functional.softmax(self.net.fd_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    fs_out = torch.nn.functional.softmax(self.net.fs_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    md_out = torch.nn.functional.softmax(self.net.md_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    ms_out = torch.nn.functional.softmax(self.net.ms_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    pr = torch.cat((fd_out, fs_out, md_out, ms_out), dim=1)
                    value, pp = torch.max(pr, 1)
                    # predicted[i]=pp.item()+1
                    if value > thresh2:

                        final_p[i] = pp.item() + 1
                    else:
                        final_p[i] = 0
            if item == 2:
                fd_out = torch.nn.functional.softmax(self.net.fs_forward(img[i:i + 1]), dim=1)
                if fd_out.data[:, 1].item() < thresh1:
                    fd_out = torch.nn.functional.softmax(self.net.fd_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    fs_out = torch.nn.functional.softmax(self.net.fs_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    md_out = torch.nn.functional.softmax(self.net.md_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    ms_out = torch.nn.functional.softmax(self.net.ms_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    pr = torch.cat((fd_out, fs_out, md_out, ms_out), dim=1)
                    value, pp = torch.max(pr, 1)
                    # predicted[i] = pp.item() + 1
                    if value > thresh2:

                        final_p[i] = pp.item() + 1
                    else:
                        final_p[i] = 0
            if item == 3:
                fd_out = torch.nn.functional.softmax(self.net.md_forward(img[i:i + 1]), dim=1)
                if fd_out.data[:, 1].item() < thresh1:
                    fd_out = torch.nn.functional.softmax(self.net.fd_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    fs_out = torch.nn.functional.softmax(self.net.fs_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    md_out = torch.nn.functional.softmax(self.net.md_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    ms_out = torch.nn.functional.softmax(self.net.ms_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    pr = torch.cat((fd_out, fs_out, md_out, ms_out), dim=1)
                    value, pp = torch.max(pr, 1)
                    # predicted[i] = pp.item() + 1
                    if value > thresh2:

                        final_p[i] = pp.item() + 1
                    else:
                        final_p[i] = 0
            if item == 4:
                fd_out = torch.nn.functional.softmax(self.net.ms_forward(img[i:i + 1]), dim=1)
                if fd_out.data[:, 1].item() < thresh1:
                    fd_out = torch.nn.functional.softmax(self.net.fd_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    fs_out = torch.nn.functional.softmax(self.net.fs_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    md_out = torch.nn.functional.softmax(self.net.md_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    ms_out = torch.nn.functional.softmax(self.net.ms_forward(img[i:i + 1]), dim=1)[:, 1:2]
                    pr = torch.cat((fd_out, fs_out, md_out, ms_out), dim=1)
                    value, pp = torch.max(pr, 1)
                    # predicted[i] = pp.item() + 1
                    if value > thresh2:
                        final_p[i] = pp.item() + 1
                    else:
                        final_p[i] = 0

        return final_p


    def eval(self,dloader,net_type='all'):
        correct = 0
        total = 0
        self.net.eval()
        with torch.no_grad():
            for data in dloader:
                images, labels, _, _ = data
                images, labels = images.to(self.device), labels.to(self.device)
                if net_type == 'all':
                    fd,fs,md,ms,outputs = self.net(images)
                elif net_type == 'fd':
                    outputs = self.net.fd_forward(images)
                elif net_type == 'fs':
                    outputs = self.net.fs_forward(images)
                elif net_type == 'md':
                    outputs = self.net.md_forward(images)
                elif net_type == 'ms':
                    outputs = self.net.ms_forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        # print('Accuracy of the network on the  images: %d %%' % (100 * correct / total))
        return  acc





