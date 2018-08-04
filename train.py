import argparse
import json
import os
from os.path import join
import random
import sys
import shutil
import time
import warnings

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

from utils import save_checkpoint, AverageMeter, accuracy

if not sys.warnoptions:
    # suppress pesky PIL EXIF warnings
    warnings.simplefilter("once")
    warnings.filterwarnings("ignore", message="(Possibly )?corrupt EXIF data.*")
    warnings.filterwarnings("ignore", message="numpy.dtype size changed.*")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed.*")


model_names = sorted(name for name in models.__dict__
                     if name[0].isupper() and not name.startswith("__")
                     and callable(models.__dict__[name]))
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
parser.add_argument('-dbg', '--dbg', action='store_true',
                    help='is debug?')

best_score = 0


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 26 epochs"""
    lr = args.lr * (0.1 ** (epoch // 26))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer,
          epoch, num_epochs, log_iter=1, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    start_time = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=175)
    for i, (images, target) in pbar:
        # measure data loading time
        data_time.update(time.time() - start_time)

        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - start_time)

        iter_num = epoch * len(train_loader) + i + 1
        if iter_num % log_iter == 0:
            logger.add_scalar('(train)loss', losses.val, iter_num)
            logger.add_scalar('(train)top1', top1.val, iter_num)
            logger.add_scalar('(train)top5', top5.val, iter_num)

        pbar.set_description(
            '[TRAIN] ep {epoch}/{num_epochs}\t'
            'loss: {loss.val:.4f} ({loss.avg:.4f})\t'
            'prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
            'fetch {data_time.val:.3f} ({data_time.avg:.3f})\t'
            '{img_sec:.2f} im/s ({img_sec_avg:.2f}))'.format(
                epoch=epoch + 1,
                num_epochs=num_epochs, loss=losses,
                top1=top1, top5=top5,
                data_time=data_time,
                img_sec=len(images) / batch_time.val,
                img_sec_avg=len(images) / batch_time.avg))

        start_time = time.time()
    logger.add_scalar('(train)avg_loss', losses.avg, epoch + 1)
    logger.add_scalar('(train)avg_top1', top1.avg, epoch + 1)
    logger.add_scalar('(train)avg_top5', top5.avg, epoch + 1)

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=180)
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in pbar:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_description(
                '[VAL] epoch {epoch}\t'
                'loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                '{img_sec:.2f} im/s ({img_sec_avg:.2f} im/s))'.format(
                    epoch=epoch + 1,
                    loss=losses,
                    top1=top1, top5=top5,
                    img_sec=len(images) / batch_time.val,
                    img_sec_avg=len(images) / batch_time.avg))

    logger.add_scalar('(train)avg_loss', losses.avg, epoch + 1)
    logger.add_scalar('(train)avg_top1', top1.avg, epoch + 1)
    logger.add_scalar('(train)avg_top5', top5.avg, epoch + 1)
    print('[VAL] epoch {}: Loss: {loss.avg:.3f} '
          'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(epoch, loss=losses, top1=top1, top5=top5))
    return top1.avg


def main():
    global args, best_score, MODELS_DIR
    args = parser.parse_args()

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

    experiment = "{}_lr{}_labels".format(args.arch, args.lr)

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

    print 'Creating train dataset...'
    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if not is_sobel:
        train_transforms.append(IMAGENET_NORMALIZE)

    train_dataset = IndexedDataset(split_dirs['train'], dataset_indices['train'],
                                   transform=transforms.Compose(train_transforms))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    print 'Creating val dataset...'
    val_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
    if not is_sobel:
        val_transforms.append(IMAGENET_NORMALIZE)

    val_dataset = IndexedDataset(split_dirs['val'], dataset_indices['val'],
                                 transform=transforms.Compose(val_transforms))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    ###############################################################################

    if args.evaluate:
        validate(val_loader, model, criterion, start_epoch)
        return

    scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)
    print 'scheduler.base_lrs=', scheduler.base_lrs
    logger.add_scalar('data/batch_size', args.batch_size, start_epoch)

    save_epoch = 50
    validate_epoch = 1
    last_lr = 100500
    for epoch in range(start_epoch, args.epochs):
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
            score = validate(val_loader, model, criterion, epoch, logger=logger)

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
