from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
from network import net
import time

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4) # you can make changes


    # Define model, Loss, and optimizer
    model = net().to(device)
    if args.initial=="False":
        model.load_state_dict(torch.load(os.path.join(args.model_path,'net_grad.ckpt')))
    #criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCELoss()
    params = list(model.linear.parameters())+list(model.bn.parameters())
    optimizer = torch.optim.Adam(params,lr=args.lr)
    m = nn.Sigmoid()

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        t1 = time.time()
        for i, data in enumerate(data_loader):

            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)

            # Forward, backward and optimize
            outputs = m(model(images))
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step,
                          loss.item()))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0: torch.save(model.state_dict(),
                    os.path.join( args.model_path, 'net_grad.ckpt')) 
        t2 = time.time() 
        print(t2 - t1) 
if __name__ == '__main__': 
    parser =argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str, default='models/', help='path for saving rainedmodels')
    parser.add_argument('--dataset_path', type=str,default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='stepsize for prining log info')
    parser.add_argument('--save_step', type=int, default=1000)
    parser.add_argument('--num_cls', type=int, default=43)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--initial',type=str,default="False")
    args = parser.parse_args()
    print(args)
main(args)
