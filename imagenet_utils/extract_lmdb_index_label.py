import argparse
import numpy as np
import os
from tensorpack.dataflow import *
from tqdm import tqdm

from data_utils.fast_dataflow import create_lmdb_stream


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to ILSVRC12 images')
    parser.add_argument('--name', choices=['train', 'val'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert args.output.endswith('.npy')

    ds, image_index = create_lmdb_stream(os.path.join(args.data, args.name + '.lmdb'), shuffle=False, return_index=True)

    index = []
    for idx, img_encoded, label in tqdm(ds.get_data(), total=len(ds)):
        idx = int(idx)
        index.append([idx, label])
    index = np.array(index)
    if not os.path.expanduser(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    assert np.array_equal(index[:, 0], np.arange(len(index)))
    with open(args.output, 'w') as f:
        np.save(f, index)
        print 'saved to {}'.format(args.output)
