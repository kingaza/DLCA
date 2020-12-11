import argparse
import os
import time
import numpy as np
import data
import shutil
from tqdm import tqdm
from importlib import import_module
from utils.log_utils import *
from utils.inference_utils import SplitComb, postprocess, plot_box 
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='ca detection')
parser.add_argument('--model', '-m', metavar='MODEL', default='model.network',
                    help='model')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input', default='', type=str, metavar='data',
                    help='directory to save images (default: none)')
parser.add_argument('--output', default='', type=str, metavar='SAVE',
                    help='directory to save prediction results(default: none)')
parser.add_argument('--test', default=1, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')

args = parser.parse_args()
print(args)


def main():
    data_name = os.path.basename(args.input)
    data_dir = os.path.dirname(args.input)
    save_dir = os.path.dirname(args.output)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')

    torch.manual_seed(0)
    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['state_dict'])
    
    cudnn.benchmark = True
    net = net.cuda()
    net = DataParallel(net)
    loss = loss.cuda()

    split_comber = SplitComb(config["split_size"],
                             config['max_stride'],
                             config['stride'],
                             config["margin"],
                             config['pad_value'])
    dataset = data.TestDataset(
        data_dir,
        data_name,
        config,
        split_comber=split_comber)
    test_loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = args.workers,
        collate_fn = data.collate,
        pin_memory=False)

    test(test_loader, net, get_pbb, save_dir, config)
    return


def test(data_loader, net, get_pbb, save_dir, config):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    net.eval()
    split_comber = data_loader.dataset.split_comber
    for idx_data, (data, coord, nzhw) in enumerate(data_loader):
        data_path = data_loader.dataset.filenames[idx_data]
        data_name = os.path.basename(data_path)
        data_dir  = os.path.dirname(data_path)
        print([idx_data,data_name])

        data = data[0][0]
        coord = coord[0][0]
        nzhw = nzhw[0]

        splitlist = range(0, len(data)+1, args.n_test)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))

        imagelist, coordlist, outputlist = [], [], []   
        for i in tqdm(range(len(splitlist)-1), desc='Predicting splits'):
            input_image = data[splitlist[i]:splitlist[i+1]]
            input_coord = coord[splitlist[i]:splitlist[i+1]]

            output = net(input_image.cuda(),input_coord.cuda())
            
            imagelist.append(input_image.numpy())
            coordlist.append(input_coord.numpy())
            outputlist.append(output.data.cpu().numpy())

        # np.savez(os.path.join(save_dir, data_name.split('.')[0]+'_net.npz'), 
        #          images=imagelist, coords=coordlist, outputs=outputlist)

        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)

        pbb, mask = get_pbb(output, thresh=-3, ismask=True)
        pbb_nms = postprocess(pbb)

        print('mask', mask)
        print('pbb_nms', pbb_nms)

        np.save(os.path.join(save_dir, data_name.split('.')[0]+'_pbb.npy'), pbb_nms)
        np.savetxt(os.path.join(save_dir, data_name.split('.')[0]+'_pbb.txt'), pbb_nms, delimiter=',', fmt='%.1f')

        plot_box(pbb_nms, data_dir, data_name, save_dir)

if __name__ == '__main__':
    main()
    
