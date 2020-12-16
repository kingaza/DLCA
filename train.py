import argparse
import os
import time
import shutil
import sys
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader

import data
from utils.log_utils import *


parser = argparse.ArgumentParser(description='ca detection')
parser.add_argument('--model', '-m', metavar='MODEL', default='model.network',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input', default='', type=str, metavar='SAVE',
                    help='directory to save train images (default: none)')
parser.add_argument('--output', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')

args = parser.parse_args()
print(args)    

if not os.path.exists(args.output):
    os.makedirs(args.output)
logfile = os.path.join(args.output, 'log')
logger = Logger(logfile)


def main():

    start_epoch = args.start_epoch
    data_dir = args.input   
    save_dir = args.output

    train_name = []
    for name in os.listdir(data_dir):
        if name.endswith("nii.gz"):
            name = name.split(".")[-3]
            train_name.append(name)

    torch.manual_seed(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1

    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))
        

    net = net.cuda()
    loss = loss.cuda()
    # cudnn.benchmark = True
    net = DataParallel(net)

    dataset = data.TrainDataset(
        data_dir,
        train_name,
        config)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    loss_total_l, loss_class_l, loss_regress_l, tpr_l, tnr_l = [], [], [], [], []

    train_hist = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, results = train_epoch(train_loader, net, loss, optimizer, epoch)
        loss_total, loss_class, loss_regress, tpr, tnr = train_metrics
        train_hist += results

        loss_total_l.append(loss_total)
        loss_class_l.append(loss_class)
        loss_regress_l.append(loss_regress)
        tpr_l.append(tpr)
        tnr_l.append(tnr)

        # update at every epoch
        plot(os.path.join(save_dir + 'train_curves.png'), 
                          loss_total_l, loss_class_l, loss_regress_l, tpr_l, tnr_l)
        np.savez(os.path.join(save_dir + 'train_curves.npz'),
                 loss_total=np.array(loss_total_l),
                 loss_class=np.array(loss_class_l),
                 loss_regress=np.array(loss_regress_l),
                 tpr=np.array(tpr_l),
                 tnr=np.array(tnr_l))

        if len(train_hist)>0:         
            df_train = pd.DataFrame(train_hist)
            df_train.to_csv(os.path.join(save_dir + 'train_info.csv'))


def get_lr(epoch):
    if epoch <= 80:    # epochs * 0.8:
        lr = args.lr
    elif epoch <= 150:  # epochs * 0.9:
        lr = 0.1 * args.lr
    elif epoch <= 250:
        lr = 0.01 * args.lr
    else:
        lr = 0.001 * args.lr        
    return lr


def train_epoch(data_loader, net, loss, optimizer, epoch):
    # start_time = time.time()
    net.train()

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    results = []

    print('='*60)    
    desc_title = 'Epoch #{} with lr={:.3e}'.format(epoch, lr)
    for data, target, coord, image_path in tqdm(data_loader, desc=desc_title):

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
        batch_result = {'epoch':epoch,}
        for i_image in range(len(image_path)):
            batch_result['image_{}'.format(i_image)] = image_path[i_image]  
            batch_result['coord_{}'.format(i_image)] = (list(coord_start)[i_image], list(coord_end)[i_image])
        batch_result['pos_true'] = 0 if loss_output[7]==0 else int(loss_output[6].detach().cpu().numpy())
        batch_result['pos_false']   = loss_output[7] - batch_result['pos_true']
        batch_result['pos_total']   = loss_output[7]
        batch_result['neg_true'] = 0 if loss_output[9]==0 else int(loss_output[8].detach().cpu().numpy())
        batch_result['neg_false']   = loss_output[9] - batch_result['neg_true']
        batch_result['neg_total']   = loss_output[9]
        batch_result['delta_z'] = loss_output[2]
        batch_result['delta_h'] = loss_output[3]
        batch_result['delta_w'] = loss_output[4]
        batch_result['delta_d'] = loss_output[5]

        results.append(batch_result)  

        # msg_fmt = "patch:{}/{}, loss:\033[1;35m{:.6f}\033[0m, class:{:.6f}, reg:{:.6f}"
        # patch_msg = msg_fmt.format(i_patch+1, len(data_loader), loss_output[0], loss_output[1], loss_output[2])
        # print(patch_msg)

        # if batch_result['pos_false'] > 0:
        #     print('There are \033[1;35m{}\033[0m false positive output.'.format(batch_result['pos_false']))
        # if batch_result['neg_false'] > 0:
        #     print('There are \033[1;35m{}\033[0m false negative output.'.format(batch_result['neg_false']))            

        # more information 
        # print('image path', image_path)
        # np_output = output.data.cpu().numpy()
        # print('output maximal', np_output[...,0].max(), 
        #       '@', np.unravel_index(np.argmax( np_output[...,0], axis=None),  
        #                             np_output[...,0].shape))        


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

    # end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    tpr = 100.0 * np.sum(metrics[:,6]) / np.sum(metrics[:,7])
    tnr = 100.0 * np.sum(metrics[:,8]) / np.sum(metrics[:,9])
    loss_total = np.mean(metrics[:,0])
    loss_class = np.mean(metrics[:,1])
    loss_regress = [np.mean(metrics[:,2]),
                    np.mean(metrics[:,3]),
                    np.mean(metrics[:,4]),
                    np.mean(metrics[:,5])]

    # print("metrics",metrics[:, 6])
    # print('Epoch {:03d} (lr {:.5f}), time {:3.2f}s'.format(epoch, lr, end_time-start_time))
    # print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
    #     100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
    #     100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
    #     np.sum(metrics[:, 7]),
    #     np.sum(metrics[:, 9]),
    #     end_time - start_time))
    # print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
    #     np.mean(metrics[:, 0]),
    #     np.mean(metrics[:, 1]),
    #     np.mean(metrics[:, 2]),
    #     np.mean(metrics[:, 3]),
    #     np.mean(metrics[:, 4]),
    #     np.mean(metrics[:, 5])))

    df_results = pd.DataFrame(results)
    df_class = df_results[['pos_true', 'pos_false', 'pos_total', 'neg_true', 'neg_false', 'neg_total']]
    df_regre = df_results[['delta_z','delta_h','delta_w','delta_d']]
    print('-'*30)
    print(df_class.sum())
    print('-'*20)
    print(df_regre.mean())

    return (loss_total,loss_class,loss_regress,tpr,tnr), results


if __name__ == '__main__':
    main()
