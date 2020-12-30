import argparse
import os
import shutil
import sys
from importlib import import_module

import numpy as np
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader

from data import TrainDataset
from utils.log_utils import Logger, plot


def get_args():

    parser = argparse.ArgumentParser(description='ca detection')
    parser.add_argument('--model', '-m', metavar='MODEL', default='model.network',
                        help='model')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=240, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=12, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-schedule', default='1', type=int, metavar='S',
                        help='learning rate decreasing type')                        
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                        help='save frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--same-dict",action="store_true",
                        help='restore all items in the same state-dict from checkpoint')    
    parser.add_argument('--input', default='', type=str, metavar='SAVE',
                        help='directory to save train images (default: none)')
    parser.add_argument('--train-dataset', default='', type=str, metavar='SAVE',
                        help='file to list train images (default: none)')   
    parser.add_argument('--valid-dataset', default='', type=str, metavar='SAVE',
                        help='file to list valid images (default: none)')      
    parser.add_argument('--aug-limit', default=0, type=float,
                        metavar='AF', help='augmentation limit')    
    parser.add_argument('--output', default='', type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')

    args = parser.parse_args()
    return args

args = get_args()
print(args) 


def get_lr(epoch):
    lr = args.lr
    if args.lr_schedule == 1:
        lr = args.lr * (args.epochs-epoch)/args.epochs 
    elif args.lr_schedule == 2:
        lr = args.lr * np.sin(np.pi/2*(args.epochs-epoch)/args.epochs)   
    elif args.lr_schedule == 3:
        lr = args.lr * np.sinc(5.*epoch/args.epochs)
    elif args.lr_schedule == 4:
        sigma_0 = args.epochs // 4
        sigma_i = sigma_0 // 6
        f_0 = norm.pdf(epoch, loc=0, scale=sigma_0) * np.sqrt(2*np.pi)*sigma_0
        f_i = np.asarray([norm.pdf(epoch, loc=sigma_0*i, scale=sigma_i) for i in range(4)])
        f_s = np.sum(f_i, axis=0) * np.sqrt(2*np.pi)*sigma_i
        lr = args.lr * (f_0 + 0.01) * (f_s + 0.1)
    return lr


def train_epoch(config, net, loss, optimizer, epoch):

    net.train()

    aug_factor = args.aug_limit * epoch / args.epochs
    dataset = TrainDataset(
        data_dir,
        train_images,
        config,
        aug_factor)
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    result_epoch = []

    print('='*60)    
    epoch_fmt = 'Epoch {}/{} training for {} exmaples with lr={:.3e} and aug_factor={:.1f}'
    print(epoch_fmt.format(epoch, args.epochs, len(data_loader), lr, aug_factor))
    for data, target, coord, image_path in tqdm(data_loader):

        data = data.cuda()
        target = target.cuda()
        coord = coord.cuda()

        output = net(data, coord)

        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

        coord_start = coord[:,:,0,0,0].detach().cpu().numpy()
        coord_end = coord[:,:,-1,-1,-1].detach().cpu().numpy()
        # print('coord: start =', coord_start, ', end =', coord_end) 

        result_batch = {'epoch':epoch,}
        for i_image in range(len(image_path)):
            result_batch['image_{}'.format(i_image)] = image_path[i_image]  
            result_batch['coord_{}'.format(i_image)] = (list(coord_start)[i_image], list(coord_end)[i_image])
        result_batch['pos_true'] = 0 if loss_output[7]==0 else int(loss_output[6])
        result_batch['pos_false']   = loss_output[7] - result_batch['pos_true']
        result_batch['pos_total']   = loss_output[7]
        result_batch['neg_true'] = 0 if loss_output[9]==0 else int(loss_output[8])
        result_batch['neg_false']   = loss_output[9] - result_batch['neg_true']
        result_batch['neg_total']   = loss_output[9]

        result_batch['loss_ce'] = loss_output[1]
        result_batch['delta_z'] = loss_output[2]
        result_batch['delta_h'] = loss_output[3]
        result_batch['delta_w'] = loss_output[4]
        result_batch['delta_d'] = loss_output[5]

        result_epoch.append(result_batch)  

    if epoch % args.save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': args.output,
            'state_dict': state_dict,
            'args': args},
            os.path.join(args.output, '%03d.ckpt' % epoch))

    metrics = np.asarray(metrics, np.float32)
    tpr = 100.0 * np.sum(metrics[:,6]) / np.sum(metrics[:,7])
    tnr = 100.0 * np.sum(metrics[:,8]) / np.sum(metrics[:,9])
    loss_total = np.mean(metrics[:,0])
    loss_class = np.mean(metrics[:,1])
    loss_regress = [np.mean(metrics[:,2]),
                    np.mean(metrics[:,3]),
                    np.mean(metrics[:,4]),
                    np.mean(metrics[:,5])]

    df_results = pd.DataFrame(result_epoch)
    df_class = df_results[['pos_true', 'pos_false', 'pos_total', 'neg_true', 'neg_false', 'neg_total']]
    df_regre = df_results[['loss_ce', 'delta_z','delta_h','delta_w','delta_d']]
    print('Training results:')
    print('-'*30)
    print(df_class.sum())
    print('-'*20)
    print(df_regre.mean())

    return (loss_total,loss_class,loss_regress,tpr,tnr), result_epoch



def test_epoch(config, net, loss, epoch):

    net.eval()

    aug_factor = args.aug_limit * epoch / args.epochs
    dataset = TrainDataset(
        data_dir,
        valid_images,
        config,
        aug_factor)
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    metrics = []
    for data, target, coord, image_path in tqdm(data_loader):

        data = data.cuda()
        target = target.cuda()
        coord = coord.cuda()

        output = net(data, coord)
        loss_output = loss(output, target)
        
        metrics.append(loss_output)

    metrics = np.asarray(metrics, np.float32)
    loss_total = np.mean(metrics[:,0])
    pos_true = int(np.sum(metrics[:,6]))
    pos_total = int(np.sum(metrics[:,7]))
    neg_true = int(np.sum(metrics[:,8]))
    neg_total = int(np.sum(metrics[:,9]))
    
    pos_margin = int(np.sum(metrics[:,10]))
    neg_margin = int(np.sum(metrics[:,11]))

    print('-'*30)
    print('Validation Results:')
    print('Total Loss={:.6f}, True Positive={:d}/{:d}, True Negative={:d}/{:d}'.format(loss_total, 
                                                             pos_true, pos_total,
                                                             neg_true, neg_total))
    print('margins: pos={}, neg={}'.format(pos_margin, neg_margin))
    return (loss_total,loss_class,loss_regress,tpr,tnr)



if __name__ == '__main__':

    data_dir = args.input   
    save_dir = args.output
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    logfile = os.path.join(args.output, 'log')
    sys.stdout = Logger(logfile)
    
    pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
    for f in pyfiles:
        shutil.copy(f, os.path.join(save_dir, f))    

    train_images = []
    valid_images = []
    if args.train_dataset:
        train_images = np.loadtxt(args.train_dataset, dtype=str)
    else:
        for name in os.listdir(data_dir):
            if name.endswith("nii.gz"):
                name = name.split(".")[-3]
                train_images.append(name)
    if args.valid_dataset:
        valid_images = np.loadtxt(args.valid_dataset, dtype=str)                


    torch.manual_seed(0)
    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    
    start_epoch = args.start_epoch
    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        
        if args.same_dict:
            print('Restore model from state-dict')
            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('Restore items only with same keys')
            ##########################################################################
            # Note: replace BatchNorm by GroupNorm
            # train with current dataset, and then do replacing with pre-trained model
            ##########################################################################
            # only update the keys in net
            net_dict = net.state_dict()
            update_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in net_dict} 

            net_dict.update(update_dict)
            net.load_state_dict(net_dict)
    else:
        if start_epoch == 0:
            start_epoch = 1


    net = net.cuda()
    loss = loss.cuda()
    # cudnn.benchmark = True
    net = DataParallel(net)
    
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)       

    loss_total_l, loss_class_l, loss_regress_l, tpr_l, tnr_l = [], [], [], [], []

    train_hist = []
    for epoch in range(start_epoch, args.epochs+1):

        train_metrics, train_results = train_epoch(config, net, loss, optimizer, epoch)
        loss_total, loss_class, loss_regress, tpr, tnr = train_metrics
        train_hist += train_results

        loss_total_l.append(loss_total)
        loss_class_l.append(loss_class)
        loss_regress_l.append(loss_regress)
        tpr_l.append(tpr)
        tnr_l.append(tnr)

        # update at every epoch
        plot(os.path.join(save_dir, 'train_curves.png'), 
                          loss_total_l, loss_class_l, loss_regress_l, tpr_l, tnr_l)
        np.savez(os.path.join(save_dir, 'train_curves.npz'),
                 loss_total=np.array(loss_total_l),
                 loss_class=np.array(loss_class_l),
                 loss_regress=np.array(loss_regress_l),
                 tpr=np.array(tpr_l),
                 tnr=np.array(tnr_l))

        if len(train_hist)>0:         
            df_train = pd.DataFrame(train_hist)
            df_train.to_csv(os.path.join(save_dir, 'train_info.csv'))
            
            
        if len(valid_images) > 0:
            with torch.no_grad():
                test_metrics = test_epoch(config, net, loss, epoch)            
