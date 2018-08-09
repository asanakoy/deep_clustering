#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark-dataflow.py

import argparse
import time
import os
from os.path import join
import json
from tqdm import tqdm

from train import create_data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', metavar='DIR',
                        default='/export/home/asanakoy/workspace/datasets/ILSVRC2012',
                        help='path to dataset')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--name', choices=['train', 'val'], default='train')
    parser.add_argument('-j', '--njobs', type=int, default=10)
    parser.add_argument('-v', '--imagenet_version', type=int, default=1, choices=[1, 2],
                        help='Images version. 1 - original, 2 - resized to 256.?')

    args = parser.parse_args()

    num_gt_classes = 1000
    split_dirs = {
        'train': join(args.data, 'train' if args.imagenet_version == 1 else 'train_256'),
        'val': join(args.data, 'val' if args.imagenet_version == 1 else 'val_256')
    }
    dataset_indices = dict()
    for key in ['train', 'val']:
        index_path = join(args.data, os.path.basename(split_dirs[key]) + '_index.json')

        if os.path.exists(index_path):
            with open(index_path) as json_file:
                dataset_indices[key] = json.load(json_file)

    assert dataset_indices['train']['class_to_idx'] == \
           dataset_indices['val']['class_to_idx']

    # TODO: ise lmdb
    train_loader_gt = create_data_loader(split_dirs['train'], dataset_indices['train'], False,
                                         sobel_normalized=True, aug='random_crop_flip',
                                         shuffle=True, num_workers=args.njobs, batch_size=args.batch)

    start = time.time()
    num_batches = 100
    i = 0
    for i, d in tqdm(enumerate(train_loader_gt), total=num_batches):
        if i == num_batches - 1:
            break
    print 'Elapsed time: {:.4f} s'.format(time.time() - start)

