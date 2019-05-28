from loader import a2d_dataset
import argparse
import sys
import os
from optparse import OptionParser
import numpy as np
import utils
from distutils.version import LooseVersion

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from cfg_a2d import train as train_cfg
from cfg_a2d import val as val_cfg
from cfg_a2d import test as test_cfg


### You can import different packages to use different networks: UNet SegNet and FCN
#from UNet import UNet
from SegNet import segnet_bn_relu
import torchfcn
import time

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
def oneHotLabel(label,cls_num,device):
    [n,h,w] = label.size()
    newLabel = torch.zeros([n,cls_num,h,w] ,dtype=torch.float32).to(device)
    
    for i in range(n):
        newLabel[i,:,:,:] = oneHotLabelHelper(label[i,:,:],cls_num,device)
    return newLabel
def oneHotLabelHelper(label,clsnum,device):
    h,w = label.size()
    result = torch.zeros((clsnum,h,w)).to(device)
    for i in range(h):
        for v in range(w):
            result[label[i,v],i,v] = 1
    return result

"""

def oneHotLabel(label,cls_num,device):
    [n,h,w] = label.size()
    label = torch.unsqueeze(label,1)
    one_hot = torch.FloatTensor(n,cls_num,h,w).zero_().to(device)
    one_hot.scatter_(1,label,1)
    return one_hot


def get_parameters(model,bias=False):
    modules_skipped = (
            nn.ReLU,
            nn.MaxPool2d,
            nn.Dropout2d,
            nn.Sequential,
            torchfcn.models.FCN32s
            )
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m,nn.ConvTranspose2d):
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module:%s'%str(m))


def cross_entropy2d(input,target,weight=None,size_average=False):
    n,c,h,w = input.size()
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        log_p = F.log_softmax(input)
    else:
        log_p = F.log_softmax(input,dim=1)
    log_p = log_p.transpose(1,2).transpose(2,3).contiguous()
    log_p = log_p[target.view(n,h,w,1).repeat(1,1,1,c)>=0]
    log_p = log_p.view(-1,c)

    mask = target>=0
    target = target[mask]
    loss = F.nll_loss(log_p,target,weight=weight,reduction='sum')
    if size_average:
        loss /=mask.data.sum()
    return loss


def train_step(args,lr=1e-1):
    #create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    train_dataset = a2d_dataset.A2DDataset(train_cfg)
    #val_dataset = a2d_dataset.A2DDataset(val_cfg)

    print("Training :",len(train_dataset))
    #print("Validation :",len(val_dataset))
    data_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)

    #Define model ,loss and optimizer
    #model = UNet(3,args.num_cls).to(device)
    model_Seg = segnet_bn_relu(3,44,True).to(device)
    model_FCN = torchfcn.models.FCN32s(args.num_cls).to(device)
    vgg16 = torchfcn.models.VGG16(pretrained=True).to(device)
    model_FCN.copy_params_from_vgg16(vgg16)

    ###You may first pretrain SegNet and FCN32s on this dataset and then do ensemble
    #model_Seg.load_state_dict(torch.load(os.path.join(args.model_path,'SegNet_NLL.ckpt')))
    #model_FCN.load_state_dict(torch.load(os.path.join(args.model_path,'FCN32s_Wm05_F.ckpt')))
    
    # for param in model_Seg.parameters():
    #     param.requires_grad = False
    # for param in model_FCN.parameters():
    #     param.requires_grad = False

    class EnNet(nn.Module):
        def __init__(self):
            super(EnNet,self).__init__()
            self.fc1 = nn.Linear(88,88)
            self.fc2 = nn.Linear(88,44)
            
        def forward(self,x1,x2):
            n,c,h,w = x1.size()
            x = torch.cat((x1,x2),1)
            #print(x.size())
            x = torch.flatten(x,start_dim=2)
            #print(x.size())
            x = torch.transpose(x,1,2)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = x.view(n,c,h,w)
            return x
    
    model_En = EnNet().to(device)
    #model_En.load_state_dict(torch.load(os.path.join(args.model_path,'FCN_Seg.ckpt')))



        
    #optimizer = optim.SGD([{'params':get_parameters(model_FCN,bias=False)},{'params':get_parameters(model_FCN,bias=True),'lr':lr*2,'weight_decay':0},],lr=lr,momentum=0.99,weight_decay=0.0005)
    optimizer = optim.Adam(model_En.parameters(),lr=lr)
    weight = torch.ones(args.num_cls).to(device)
    weight[0] = 0.5

    #criterion = nn.NLLLoss(weight=weight)
    #m = nn.LogSoftmax()

    #Train the model
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        t1 = time.time()
        label_trues,label_preds = [],[]

        for i ,data in enumerate(data_loader):

            #mini-batch
            imgs = data[0].to(device)
            true_masks = data[1].to(device)
            #label_trues.append(ture_masks.cpu().numpy())
            
            #true_masks_oh = oneHotLabel(true_masks,args.num_cls,device)
            
            
            masks_pre_Seg = model_Seg(imgs)
            masks_pre_FCN = model_FCN(imgs)
            masks_pre = model_En(masks_pre_Seg,masks_pre_FCN)

            
            
            loss = cross_entropy2d(masks_pre,true_masks,weight=weight)
            model_En.zero_grad()
            loss.backward()
            optimizer.step()

            #Log info 
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch,args.num_epochs,i,total_step,loss.item()))
                #filename = 'lossdata.txt'
                #with open("".join([args.losspath,filename]),'a') as f:
                    #f.write(str(loss.item())+"\n")
            
            #save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(model_En.state_dict(),os.path.join(
                    args.model_path,'FCN_SegNet_En.ckpt'
                ))
        t2 = time.time()
        print('Epoch training time:',t2-t1)
        
        #metrics = utils.label_accuracy_score(label_trues,label_preds,n_class=args.num_cls)
        #metrics = np.array(metrics)
        #metrics *= 100
        #print('''\
         #       Accuracy:{0}
          #       Accuracy Class: {1}
           #     Mean IU: {2}
            #    FWAV Accuracy: {3}'''.format(*metrics))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=100, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=44)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--losspath',type=str,default="../lossdata/",help="path to saving the loss data in training process")

    args = parser.parse_args()
    print(args)
train_step(args)
            



            


            
            

                

              


            
