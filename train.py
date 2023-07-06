import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

from dataset.CVUSA import CVUSA
from models.feature_extractor import FeatureExtractor
from models.siamese_network import SiameseNet
from criterion.soft_triplet import SoftTripletBiLoss
from dataset.global_sampler import DistributedMiningSampler
from utils.meters import AverageMeter, ProgressMeter
from utils.learning_rate import adjust_learning_rate, get_lr
from utils.metrics import accuracy
from utils.save import save_checkpoint
from utils.models import modify_model, copy_weights

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch image based localization \
    training')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dim', default=1024, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--mining', action='store_true',
                    help='mining')
parser.add_argument('--ground-color-space', default='RGB', type=str,
                    help='color space to use for ground images available \
                        options: `RGB`, `L`)')
parser.add_argument('--aerial-color-space', default='RGB', type=str,
                    help='color space to use for aerial images (available \
                        options: `RGB`, `L`)')
parser.add_argument('--data-dir', default='/groups/amahalan/NatesData/CVUSA/', 
                    type=str,
                    help='root directory containing the CVUSA dataset')
parser.add_argument('--resume-from', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ground-net-weights', default='', type=str, metavar='PATH',
                    help='path to desired checkpoint (default: none)')

best_top1_accuracy = 0

def main():
    args = parser.parse_args()
    print(args)
    global best_top1_accuracy
    # initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9999'
    if not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend, 
                                world_size=args.world_size, 
                                rank=args.rank)
    # create model
    model = SiameseNet()
    # model = modify_model(model, args)
    print(model)
    torch.cuda.set_device(args.gpu)
    model.cuda()
    model = DDP(model, device_ids=[args.gpu])
    # copy weights from pre-trained RGB ground net to grayscale ground net
    if args.ground_net_weights:
        print("=> Initializing ground net with weights from checkpoint \
                '{}'".format(args.ground_net_weights))
        copy_weights(model, args)
    # create train and val dataloaders
    train_dataset = CVUSA(root=args.data_dir, 
                          mode='train', 
                          ground_color_space=args.ground_color_space,
                          aerial_color_space=args.aerial_color_space)
    val_ground_dataset = CVUSA(root=args.data_dir, 
                               mode='test_ground', 
                               ground_color_space=args.ground_color_space)
    val_aerial_dataset = CVUSA(root=args.data_dir, 
                               mode='test_aerial', 
                               aerial_color_space=args.aerial_color_space)
    train_sampler = DistributedMiningSampler(train_dataset, 
                                             batch_size=args.batch_size, 
                                             dim=args.dim, 
                                             save_path=args.save_path)
    # if resuming training, load last saved sampler queue and counter
    if args.resume_from:
        train_sampler.load(args.resume_from.replace(args.resume_from.\
            split('/')[-1],''))
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=(train_sampler is None),
                              num_workers=args.workers, 
                              pin_memory=True, 
                              sampler=train_sampler, 
                              drop_last=True)
    val_ground_loader = DataLoader(val_ground_dataset, 
                                   batch_size=args.batch_size, 
                                   shuffle=False,
                                   num_workers=args.workers, 
                                   pin_memory=True)
    val_aerial_loader = DataLoader(val_aerial_dataset, 
                                   batch_size=args.batch_size, 
                                   shuffle=False,
                                   num_workers=args.workers, 
                                   pin_memory=True)

    # create soft-margin triplet loss
    criterion = SoftTripletBiLoss().cuda()
    # create SGD optimizer
    optimizer = SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    # load saved checkpoint if resuming training
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume_from)
            args.start_epoch = checkpoint['epoch']
            best_top1_accuracy = checkpoint['best_top1_accuracy']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".
                  format(args.resume_from, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    for epoch in range(args.start_epoch, args.epochs):
        
        print('start epoch:{}, date:{}'.format(epoch, datetime.now()))
        train_sampler.set_epoch(epoch)
        train_sampler.update_epoch()
        adjust_learning_rate(optimizer, epoch, args)
        # train
        train(train_loader, model, criterion, optimizer, epoch, args, 
              train_sampler)
        # evaluate on validation set
        top1_accuracy = validate(val_ground_loader, val_aerial_loader, 
                                 model, args)
        # keep track of best accuracy
        is_best = top1_accuracy > best_top1_accuracy
        best_top1_accuracy = max(top1_accuracy, best_top1_accuracy)
        # save checkpoint
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_top1_accuracy': best_top1_accuracy,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename='checkpoint.pth.tar', args=args)

def train(train_loader, 
          model, 
          criterion, 
          optimizer, 
          epoch=None, 
          args=None, 
          train_sampler=None):
    
    # initialize meters to keep track of time-varying variables
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mean_ps = AverageMeter('Mean-P', ':6.2f')
    mean_ns = AverageMeter('Mean-N', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mean_ps, mean_ns],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()
    
    end = time.time()

    for batch_idx, (ground_images, aerial_images, indexes) in \
        enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.ground_color_space == 'L':
            ground_images = torch.cat([ground_images] * 3, dim=1)
        ground_images = ground_images.cuda()
        aerial_images = aerial_images.cuda()
        indexes = indexes.cuda()
        ground_embedding, aerial_embedding = model(ground_images, 
                                                   aerial_images)
        loss, mean_pos_sim, mean_neg_sim = criterion(ground_embedding, 
                                                     aerial_embedding)
        train_sampler.update(aerial_embedding.detach().cpu().numpy(),
                             ground_embedding.detach().cpu().numpy(),
                             indexes.detach().cpu().numpy())
        # update meters with current values 
        losses.update(loss.item(), ground_images.size(0))
        mean_ps.update(mean_pos_sim, ground_images.size(0))
        mean_ns.update(mean_neg_sim, ground_images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # display progress 
        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

        del loss
        del ground_embedding
        del aerial_embedding

def validate(val_ground_loader, val_aerial_loader, model, args):
    
    batch_time = AverageMeter('Time', ':6.3f')
    progress_ground = ProgressMeter(
        len(val_ground_loader),
        [batch_time],
        prefix='Test_ground: ')
    progress_aerial = ProgressMeter(
        len(val_aerial_loader),
        [batch_time],
        prefix='Test_aerial: ')

    # switch to evaluate mode
    ground_model = model.module.ground_embedding.cuda()
    aerial_model = model.module.aerial_embedding.cuda()
    ground_model.eval()
    aerial_model.eval()

    ground_embeddings = np.zeros([len(val_ground_loader.dataset), args.dim])
    ground_labels = np.zeros([len(val_ground_loader.dataset)])
    aerial_embeddings = np.zeros([len(val_aerial_loader.dataset), args.dim])
    
    with torch.no_grad():
        end = time.time()
        # compute all aerial image embeddgins
        for batch_idx, (aerial_images, indexes) in enumerate(val_aerial_loader):
            aerial_images = aerial_images.cuda()
            indexes = indexes.cuda()
            # compute output
            batch_aerial_embedding = aerial_model(aerial_images)
            aerial_embeddings[indexes.cpu().numpy().astype(int), :] = \
            batch_aerial_embedding.detach().cpu().numpy()
            # measure elapsed time since batch data was loaded
            batch_time.update(time.time() - end)
            end = time.time()
            # display validation progress
            if batch_idx % args.print_freq == 0:
                progress_aerial.display(batch_idx)            
        end = time.time()
        # compute all ground image embeddings
        for batch_idx, (ground_images, indexes, labels) in \
            enumerate(val_ground_loader):
            if args.ground_color_space == 'L':
                ground_images = torch.cat([ground_images] * 3, dim=1)
            ground_images = ground_images.cuda()
            indexes = indexes.cuda()
            labels = labels.cuda()
            # compute output
            batch_ground_embedding = ground_model(ground_images)
            ground_embeddings[indexes.cpu().numpy(), :] = \
            batch_ground_embedding.cpu().numpy()
            ground_labels[indexes.cpu().numpy()] = labels.cpu().numpy()
            # measure elapsed time since batch data was loaded
            batch_time.update(time.time() - end)
            end = time.time()
            # display validation progress
            if batch_idx % args.print_freq == 0:
                progress_ground.display(batch_idx)   
        # compute top-k accuracy
        [top1, top5] = accuracy(ground_embeddings, 
                                aerial_embeddings, 
                                ground_labels.astype(int))
    
    if args.evaluate:
        np.save(os.path.join(args.save_path, 'ground_embeddings.npy'), 
                ground_embeddings)
        np.save(os.path.join(args.save_path, 'aerial_embeddings.npy'), 
                aerial_embeddings)
        
    return top1

if __name__ == '__main__':
    main()


