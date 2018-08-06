import os
from os.path import join
import time
import torch
import torch.nn as nn
import torch.optim
import torchvision
import torch.utils.data
from tqdm import tqdm

from utils import save_checkpoint, AverageMeter, accuracy


def train(train_loader, model, criterion, optimizer,
          epoch, num_epochs, log_iter=1, logger=None, tag='train'):
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
            logger.add_scalar('({})loss'.format(tag), losses.val, iter_num)
            logger.add_scalar('({})top1'.format(tag), top1.val, iter_num)
            logger.add_scalar('({})top5'.format(tag), top5.val, iter_num)

        pbar.set_description(
            '[{tag}] ep {epoch}/{num_epochs}\t'
            'loss: {loss.val:.4f} ({loss.avg:.4f})\t'
            'prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
            'fetch {data_time.val:.3f} ({data_time.avg:.3f})\t'
            '{img_sec:.2f} im/s ({img_sec_avg:.2f}))'.format(
                tag=tag.upper(),
                epoch=epoch + 1,
                num_epochs=num_epochs, loss=losses,
                top1=top1, top5=top5,
                data_time=data_time,
                img_sec=len(images) / batch_time.val,
                img_sec_avg=len(images) / batch_time.avg))

        start_time = time.time()
    logger.add_scalar('({})avg_loss'.format(tag), losses.avg, epoch + 1)
    logger.add_scalar('({})avg_top1'.format(tag), top1.avg, epoch + 1)
    logger.add_scalar('({})avg_top5'.format(tag), top5.avg, epoch + 1)

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, epoch, logger, tag='val'):
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
                '[{tag}] epoch {epoch}\t'
                'loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                '{img_sec:.2f} im/s ({img_sec_avg:.2f} im/s))'.format(
                    tag=tag.upper(),
                    epoch=epoch + 1,
                    loss=losses,
                    top1=top1, top5=top5,
                    img_sec=len(images) / batch_time.val,
                    img_sec_avg=len(images) / batch_time.avg))

    logger.add_scalar('({})avg_loss'.format(tag), losses.avg, epoch + 1)
    logger.add_scalar('({})avg_top1'.format(tag), top1.avg, epoch + 1)
    logger.add_scalar('({})avg_top5'.format(tag), top5.avg, epoch + 1)
    print('[{tag}] epoch {epoch}: Loss: {loss.avg:.3f} '
          'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(tag=tag.upper(), epoch=epoch + 1, loss=losses, top1=top1, top5=top5))
    return top1.avg