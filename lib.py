import numpy as np
import os
from os.path import join
import time
import torch
import torch.nn as nn
import torch.optim
import torchvision
import torch.utils.data
from tqdm import tqdm

from models.alexnet import AlexNetTruncated, AlexNetLinear
from utils import save_checkpoint, AverageMeter, accuracy, timed_operation
from data_utils.fast_dataflow import TorchBatchData


def train(train_loader, model, criterion, optimizer,
          epoch, num_epochs, log_iter=1, logger=None, tag='train'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    start_time = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                ncols=175, desc='[{tag}]'.format(tag=tag.upper()))
    for i, (images, target) in pbar:
        # measure data loading time
        data_time.update(time.time() - start_time)

        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))  # returns tensors!
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

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

    pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                ncols=180, desc='[{tag}]'.format(tag=tag.upper()))
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
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

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


def validate_gt_linear(train_loader_gt, val_loader_gt, num_gt_classes, net, layer_name, criterion, epoch, lr=0.01, num_train_epochs=2, logger=None, tag='VAL_GT'):
    """
    Train a linear classifier on top of conv4 features and evaluate using GT labels.
    """
    assert num_train_epochs > 0, 'validate_gt_linear: num_train_epochs must be > 0'
    net = nn.DataParallel(AlexNetLinear(net.module, layer_name, num_classes=num_gt_classes)).cuda()

    optimizer = torch.optim.SGD(net.module.linear.parameters(), lr,
                                momentum=0.9,
                                weight_decay=0.0005)

    weight_sobel = net.module._modules['base_net']._modules['layers']._modules['sobel'].weight.data.cpu().numpy().copy()
    weight_conv1 = net.module._modules['base_net']._modules['layers']._modules['0'].weight.data.cpu().numpy().copy()
    print 'Batch size:', train_loader_gt.batch_size

    for epoch in range(num_train_epochs):
        train(train_loader_gt, net, criterion, optimizer,
              epoch, num_train_epochs,
              log_iter=100, logger=logger, tag='train({})'.format(tag))

        weight_sobel_after = net.module._modules['base_net']._modules['layers']._modules['sobel'].weight.data.cpu().numpy().copy()
        weight_conv1_after = net.module._modules['base_net']._modules['layers']._modules['0'].weight.data.cpu().numpy().copy()

        assert np.allclose(weight_sobel, weight_sobel_after), 'Sobel weights changed!'
        assert np.allclose(weight_conv1, weight_conv1_after), 'conv1 weights changed!'

        acc = validate(val_loader_gt, net, criterion, epoch, logger, tag='val_gt_linear')
        print '[{}] Prec@1 {}'.format(tag.upper(), acc)
    return acc


def extract_features(data_loader, net, layer_name):
    """
    Extract features from the specific layer
    Args:
        data_loader:
        net:
        layer_name:

    Returns:

    """
    net_trunc = AlexNetTruncated(net.module, layer_name).cuda()
    net_trunc.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    assert isinstance(data_loader, TorchBatchData) or isinstance(data_loader.sampler, torch.utils.data.SequentialSampler), 'Data must be sequential!'
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), ncols=180, desc='[FEATS]')

    indices = None
    features = None
    cur_pos = 0

    with torch.no_grad():
        start_time = time.time()
        for i, (images, target, cur_indexes) in pbar:
            data_time.update(time.time() - start_time)

            images = images.cuda(non_blocking=True)
            output = net_trunc(images)
            batch_time.update(time.time() - start_time)

            if features is None:
                features_shape = (len(data_loader.dataset), np.prod(output.shape[1:]))
                print '\nMemory allocation for features (shape={})...'.format(features_shape)
                features = np.zeros(features_shape, dtype=np.float32)
                indices = np.zeros(features_shape[0], dtype=np.int32)
            features[cur_pos:cur_pos + len(output), ...] = output
            indices[cur_pos:cur_pos + len(output)] = cur_indexes
            cur_pos += len(output)

            pbar.set_description(
                '[FEATS] \t'
                'fetch {data_time.val:.3f} ({data_time.avg:.3f})\t'
                '{img_sec:.2f} im/s ({img_sec_avg:.2f} im/s))'.format(
                data_time=data_time,
                img_sec=len(images) / batch_time.val,
                img_sec_avg=len(images) / batch_time.avg))

            start_time = time.time()
    assert cur_pos == len(features)
    with timed_operation('Permute features in the appropriate order...', log_start=True, tformat='m'):
        features = features[indices]
    return features
