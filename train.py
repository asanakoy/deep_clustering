import argparse
import json
import numpy as np
import os
from os.path import join
import random
import sys
import shutil
import time
import warnings
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import models
from data_utils.index_imagenet import index_imagenet
from data_utils.index_imagenet import IndexedDataset
from data_utils.transforms import IMAGENET_NORMALIZE, SIMPLE_NORMALIZE

from lib import train, validate, validate_gt_linear, extract_features
# import unsupervised.faissext
from utils import save_checkpoint, AverageMeter, accuracy

if not sys.warnoptions:
    # suppress pesky PIL EXIF warnings
    warnings.simplefilter("once")
    warnings.filterwarnings("ignore", message="(Possibly )?corrupt EXIF data.*")
    warnings.filterwarnings("ignore", message="numpy.dtype size changed.*")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed.*")


# model_names = sorted(name for name in models.__dict__
#                      if name[0].isupper() and not name.startswith("__")
#                      and callable(models.__dict__[name]))
model_names = ['AlexNet', 'AlexNetSobel']

MODELS_DIR = './results'

parser = argparse.ArgumentParser(description='DeepClustering Training')
parser.add_argument('-data', metavar='DIR',
                    default='/export/home/asanakoy/workspace/datasets/ILSVRC2012',
                    help='path to dataset')
parser.add_argument('-o', '--output_dir', default=None, help='output dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='AlexNet',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: AlexNet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--decay_step', type=float, default=26, metavar='EPOCHS',
                    help='learning rate decay step')
parser.add_argument('--decay_gamma', type=float, default=0.1,
                    help='learning rate decay coeeficient')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 0.0005)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-suf', '--exp_suffix', default='', help='experiment suffix')
parser.add_argument('-unsup', '--unsupervised', action='store_true',
                    help='is unsupervised training?')
parser.add_argument('-recluster_epoch', '--recluster_epoch', type=int, default=1,
                    help='Recluster every K epoch.')
parser.add_argument('-nc', '--num_clusters', type=int, default=10000,
                    help='Number of clusters for unsupervised training.')
parser.add_argument('-l', '--clustering_layer', default='fc7',
                    help='which layer to extract features for clustering?')
parser.add_argument('-eval_layer', '--eval_layer', default='conv3',
                    help='which layer to use for training a linear classifier (only for unsupervised training)')

parser.add_argument('-dbg', '--dbg', action='store_true',
                    help='is debug?')

best_score = 0


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 26 epochs"""
    lr = args.lr * (0.1 ** (epoch // 26))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def collate_concat(batch):
    """Puts each data field into a tensor with outer dimension batch size
       Can be used when each element produced by the transformer (or dataset) is already a group of several items.
       In this case we concatenate tensors and do not create an extra axis.
    """
    if isinstance(batch[0], torch.Tensor):
        out = None
        if torch.utils.data.dataloader._use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.cat(batch, dim=0, out=out)
    else:
        raise NotImplementedError()


def create_data_loader(split_dir, dataset_index, is_sobel, aug='central_crop', shuffle=False,
                       num_workers=2, return_index=False):
    print 'Creating train dataset... ({})'.format(aug)
    if aug == 'random_crop_flip':
        transf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        collate_fn = torch.utils.data.dataloader.default_collate
    elif aug == 'central_crop':
        transf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
        collate_fn = torch.utils.data.dataloader.default_collate
    elif aug == '10_crop':
        to_tensor = transforms.ToTensor()
        transf = [
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([to_tensor(crop) for crop in crops])),  # returns a 4D tensor
        ]
        collate_fn = collate_concat
    else:
        raise ValueError('Unknown aug:' + aug)

    if not is_sobel:
        if aug != '10_crop':
            transf.append(IMAGENET_NORMALIZE)
        else:
            transf.append(transforms.Lambda(lambda crops: torch.stack([IMAGENET_NORMALIZE(crop) for crop in crops])))
    else:
        # NOTE: May be use some other kind of normalization before sobel?
        pass

    dataset = IndexedDataset(split_dir, dataset_index,
                             transform=transforms.Compose(transf), return_index=return_index)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn)
    return loader


def unsupervised_clustering_step(cur_epoch, model, is_sobel, split_dirs, dataset_indices, num_workers, labels_holder, logger):
    if args.unsupervised:
        print '[TRAIN]...'
        train_loader = create_data_loader(split_dirs['train'], dataset_indices['train'], is_sobel, aug='central_crop',
                                          shuffle=False, num_workers=num_workers, return_index=True)
        features = extract_features(train_loader, model, args.clustering_layer)

        labels = unsupervised.faissext.do_clustering(features, args.num_clusters)

        # Evaluate NMI
        if 'labels_gt' not in labels_holder:
            labels_holder['labels_gt'] = zip(*dataset_indices['train']['samples'])[0]
        nmi = normalized_mutual_info_score(labels_holder['labels_gt'], labels)
        print 'NMI t / GT = {:.4f}'.format(nmi)
        logger.add_scalar('NMI', nmi, cur_epoch)

        if 'labels_prev_step' in labels_holder:
            nmi = normalized_mutual_info_score(labels_holder['labels_prev_step'], labels)
            print 'NMI t / t-1 = {:.4f}'.format(nmi)
            logger.add_scalar('NMI_t-1', nmi, cur_epoch)
        labels_holder['labels_prev_step'] = labels

        dataset_indices['train_unsupervised'] = {
            'classes': np.arange(args.num_clusters),
            'class_to_idx': {i: i for i in xrange(args.num_clusters)},
            'samples': [(sample[0], lbl) for sample, lbl in zip(dataset_indices['train_unsupervised']['train'], labels)]
        }

        train_loader = create_data_loader(split_dirs['train'], dataset_indices['train_unsupervised'], is_sobel, aug='random_crop_flip',
                                          shuffle=True, num_workers=num_workers)
        return train_loader


def main():
    global args, best_score, MODELS_DIR
    args = parser.parse_args()
    print args

    if args.dbg:
        MODELS_DIR = join(MODELS_DIR, 'dbg')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # if torch.cuda.is_available() and not args.cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    cudnn.benchmark = True

    is_sobel = args.arch.endswith('Sobel')
    print 'is_sobel', is_sobel

    if args.pretrained:
        assert False, 'Not supported for now'
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    experiment = "{}_lr{}_{}".format(args.arch, args.lr, 'unsup' if args.unsupervised else 'labels')
    if args.unsupervised:
        experiment += '_nc{nc}_l{clustering_layer}_rec{rec_epoch}'.format(nc=args.number_clusters,
                                                                          clustering_layer=args.clustering_layer,
                                                                          rec_epoch=args.recluster_epoch)

    checkpoint = None
    if args.output_dir is None:
        args.output_dir = join(MODELS_DIR, experiment + '_' + args.exp_suffix)

    if args.output_dir is not None and os.path.exists(args.output_dir):
        ckpt_path = join(args.output_dir, 'checkpoint.pth.tar')
        if not os.path.isfile(ckpt_path):
            print "=> no checkpoint found at '{}'\nUsing model_best.pth.tar".format(ckpt_path)
            ckpt_path = join(args.output_dir, 'model_best.pth.tar')
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        else:
            print "=> no checkpoint found at '{}'".format(ckpt_path)
            ans = None
            while ans != 'y' and ans != 'n':
                ans = raw_input('Clear the dir {}? [y/n] '.format(args.output_dir)).lower()
            if ans.lower() == 'y':
                shutil.rmtree(args.output_dir)
            else:
                print 'Just write in the same dir.'
                # raise IOError("=> no checkpoint found at '{}'".format(ckpt_path))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        if 'best_score' in checkpoint:
            best_score = checkpoint['best_score']
        else:
            print 'WARNING! NO best "score_found" in checkpoint!'
            best_score = 0
        print 'Best score:', best_score
        print 'Current score:', checkpoint['cur_score']
        model.load_state_dict(checkpoint['state_dict'])
        print 'state dict loaded'
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            param_group['initial_lr'] = args.lr
    else:
        start_epoch = 0
        best_score = 0

    logger = SummaryWriter(log_dir=args.output_dir)

    ### Data loading ###
    split_dirs = dict()
    dataset_indices = dict()
    for key in ['train', 'val']:
        split_dirs[key] = join(args.data, key)
        index_path = join(args.data, key + '_index.json')

        if os.path.exists(index_path):
            with open(index_path) as json_file:
                dataset_indices[key] = json.load(json_file)
        else:
            print 'Indexing ' + key
            dataset_indices[key] = index_imagenet(split_dirs[key], index_path)

    assert dataset_indices['train']['class_to_idx'] == \
           dataset_indices['val']['class_to_idx']

    num_workers = args.workers if args.unsupervised else max(1, args.workers / 2)

    print '[TRAIN]...'
    if args.unsupervised:
        train_loader_gt = create_data_loader(split_dirs['train'], dataset_indices['train'], is_sobel, aug='random_crop_flip',
                                          shuffle=True, num_workers=num_workers)
        eval_gt_aug = '10_crop'
        val_loader_gt = create_data_loader(split_dirs['val'], dataset_indices['val'], is_sobel, aug=eval_gt_aug,
                                           shuffle=False, num_workers=num_workers)
    else:
        train_loader = create_data_loader(split_dirs['train'], dataset_indices['train'], is_sobel, aug='random_crop_flip',
                                          shuffle=True, num_workers=num_workers)
        print '[VAL]...'
        # with GT labels!
        val_loader = create_data_loader(split_dirs['val'], dataset_indices['val'], is_sobel, aug='central_crop',
                                        shuffle=False, num_workers=num_workers)
    ###############################################################################

    scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)
    print 'scheduler.base_lrs=', scheduler.base_lrs
    logger.add_scalar('data/batch_size', args.batch_size, start_epoch)

    save_epoch = 50
    if not args.unsupervised:
        validate_epoch = 1
    else:
        validate_epoch = 10
        labels_holder = {}  # utility container to save labels from the previous clustering step

    last_lr = 100500
    for epoch in range(start_epoch, args.epochs):
        if args.unsupervised and (epoch == start_epoch or epoch % args.recluster_epoch == 0):
            train_loader = unsupervised_clustering_step(cur_epoch, model, is_sobel, split_dirs, dataset_indices,
                                                        num_workers, labels_holder, logger)
        scheduler.step(epoch=epoch)
        if last_lr != scheduler.get_lr()[0]:
            last_lr = scheduler.get_lr()[0]
            print 'LR := {}'.format(last_lr)
        logger.add_scalar('data/lr', scheduler.get_lr()[0], epoch)
        logger.add_scalar('data/weight_decay', args.weight_decay, epoch)

        top1_avg, top5_avg, loss_avg = \
            train(train_loader, model, criterion, optimizer,
                  epoch, args.epochs,
                  log_iter=100, logger=logger)

        if (epoch + 1) % validate_epoch == 0:
            # evaluate on validation set
            if not args.unsupervised:
                score = validate(val_loader, model, criterion, epoch, logger=logger)
            else:
                score = validate_gt_linear(train_loader_gt, val_loader_gt, net, args.eval_layer, criterion, epoch, lr=0.01, num_train_epochs=2,
                                           logger=None, tag='val_gt_{}_{}'.format(args.eval_layer, eval_gt_aug))

            # remember best prec@1 and save checkpoint
            is_best = score > best_score
            best_score = max(score, best_score)
        else:
            is_best = False

        if (epoch + 1) % save_epoch == 0:
            filepath = join(args.output_dir, 'checkpoint-{:05d}.pth.tar'.format(epoch + 1))
        else:
            filepath = join(args.output_dir, 'checkpoint.pth.tar')
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'cur_score': score,
            'best_score': best_score,
            'top1_avg_accuracy_train': top1_avg,
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, filepath=filepath)


if __name__ == '__main__':
    main()
