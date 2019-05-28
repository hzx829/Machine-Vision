from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import h5py
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader

from cfg_a2d import train as train_cfg
from cfg_a2d import val as val_cfg
from cfg_a2d import test as test_cfg
import pickle

import utils
from SegNet import segnet_bn_relu
from EnNet import EnNet
#from fcn32s import FCN32s
#from UNet import UNet
#from ResFCN import FCN
import torchfcn
import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def oneHot2One(output,device):#output:[classes,w,h]
    print("output:",output.size())
    _,w,h = output.size()
    result = torch.zeros((w,h)).to(device)
    for i in range(w):
        for v in range(h):
            vertical = output[:,i,v]
            result[i,v] = torch.argmax(vertical)
    print("result:",result.size())
    print('pred:',np.count_nonzero(result.cpu().numpy()))
    return result#result:[224,224]

def predict(args):
    test_dataset = a2d_dataset.A2DTestDataset(test_cfg) 
    #val_dataset = a2d_dataset.A2DDataset(val_cfg)
    data_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4)

    
    #model_Seg = segnet_bn_relu(3,44).to(device)
    #vgg_model = VGGNet(requires_grad=False).to(device)
    model_FCN = torchfcn.models.FCN32s(n_class=44).to(device)
    #m = nn.LogSoftmax()
    #model_Seg.load_state_dict(torch.load(os.path.join(args.model_path,'SegNet_NLL.ckpt')))
    model_FCN.load_state_dict(torch.load(os.path.join(args.model_path,'FCN32s_Wm05_F.ckpt')))
    
    #model_En = EnNet().to(device)
    #model_En.load_state_dict(torch.load(os.path.join(args.model_path,'FCN_Seg.ckpt')))
    
    
    gt_list = []
    mask_list = []
    acc = 0
    iu = 0
    #model_En.eval()
    model_FCN.eval()
            
    with torch.no_grad():

        for batch_idx,data in enumerate(data_loader):

            print("step:{}/{}".format(batch_idx,len(test_dataset)))
            #images = data[0].to(device)
            #test
            images = data[0].unsqueeze(0).to(device)
            #gt = data[1].to(device).cpu().numpy()#[224,224]
            #print('gt:',np.count_nonzero(gt.cpu().numpy()))
            #output = model(images)
            #mask = oneHot2One(output,device)#[224,224]
            #mask_Seg = model_Seg(images)
            output = model_FCN(images)
            #output = model_En(mask_Seg,mask_FCN)
            
            mask = output.data.max(1)[1].cpu().numpy().astype(np.uint8)[:,:,:]
            mask_list.append(mask)
            #if batch_idx == 10:
                #break
            #gt_list.append(gt)

            #metrics = utils.label_accuracy_score(gt.unsqueeze(0).cpu().numpy().astype(np.uint8),mask.unsqueeze(0).cpu().numpy().astype(np.uint8),n_class=args.num_cls)
            #metrics = np.array(metrics)
            #metrics *= 100
            #acc +=metrics[1]
            #iu +=metrics[2]
            #print('''\
            #        Accuracy: {0}
            #        Accuracy_cls: {1}
             #       Mean IU: {2}
             #       FWAV Accuracy: {3}'''.format(*metrics))
            
    with open(args.filename, 'wb') as f:
        pickle.dump(mask_list,f)
    #with open('train_gt.pkl','wb') as f:
        #pickle.dump(gt_list,f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=44)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--filename',type=str,default="")
    args = parser.parse_args()
    print(args)
predict(args)
            



