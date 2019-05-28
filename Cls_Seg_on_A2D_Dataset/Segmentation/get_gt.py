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

import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_gt(args):
    
    val_dataset = a2d_dataset.A2DDataset(val_cfg,args.dataset_path)
    data_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1)

    mask_list = []
    model = Unet(3,44).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_path,'FCN_SegNet.ckpt')))#load your model

    model.eval()
    with torch.no_grad():
        for batch_idx,data in enumerate(data_loader):
            images = data[0].to(device).unsqueeze(0)
            output = model(images)
            mask = oneHot2One(output,device).cpu().detach().numpy()
            mask = mask.astype(np.uint8)
            mask_list.append(mask)
    with open('FCN_SegNet.pkl', 'wb') as f:
        pickle.dump(mask_list,f)
    
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
    args = parser.parse_args()
    print(args)
predict(args)