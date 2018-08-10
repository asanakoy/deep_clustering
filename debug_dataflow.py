import collections
import numpy as np
import pprint
import re
import six
from tensorpack.dataflow import *
from tensorpack.dataflow.image import AugmentorList
from tensorpack.utils import logger
import torch
from torchvision import transforms
from tqdm import tqdm
import cv2

from data_utils.transforms import pil_to_np_array, IMAGENET_NORMALIZE_NP
from data_utils.fast_dataflow import TorchBatchData, TorchAugmentorList

_use_shared_memory = False


def create_lmdb_stream(lmdb_path, shuffle=False, new_labels=None, return_index=False):
    from tensorpack.utils.serialize import loads
    ds = LMDBData(lmdb_path, shuffle=shuffle)
    if shuffle:
        logger.warning('Performance may suffer when loading LMDB with shuffle=True.')

    def desearialize_data_point(dp):
        idx = int(dp[0])  # index from 0 to N
        value = loads(dp[1])
        assert len(value) == 1, len(value)
        [img_id] = value

        if return_index:
            return img_id, idx
        else:
            return img_id

    return MapData(ds, desearialize_data_point)


def create_fast_lmdb_flow(lmdb_path, nr_proc=10, batch_size=256, shuffle=False, return_index=False):
    # image_index = 0
    # ds = LMDBSerializer.load(lmdb_path, shuffle=shuffle)
    ds = create_lmdb_stream(lmdb_path, shuffle=shuffle, return_index=return_index)
    # ds = LocallyShuffleData(ds, buffer_size=10000)
    # We use PrefetchData to launch the base LMDB Flow in only one process,
    # and only parallelize the transformations with another
    ds = PrefetchData(ds, nr_prefetch=5000, nr_proc=1)
    ds = MapDataComponent(ds, lambda x: x, index=0)
    transform = transforms.Compose([
        lambda x: x
    ])
    ds = AugmentImageComponent(ds, TorchAugmentorList(transform), index=0, copy=False)
    ds = PrefetchDataZMQ(ds, nr_proc=nr_proc)
    ds = TorchBatchData(ds, batch_size=batch_size, remainder=True)
    return ds


if __name__ == '__main__':
    lmdb_path = '/export/home/asanakoy/workspace/tmp/dummy_1M.lmdb'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--njobs', type=int, default=10)
    args = parser.parse_args()
    ds = create_fast_lmdb_flow(lmdb_path, nr_proc=args.njobs, batch_size=256, shuffle=False, return_index=True)
    print 'len(ds):', len(ds)

    indices = []
    indices_set = set()
    for img_id, idxb in tqdm(ds):
        assert img_id == idxb, '{} != {}'.format(img_id, idxb)
        indices.append(idxb.tolist())
        for idx in idxb:
            if idx in indices_set:
                print 'Warning idx={} is already in the set'.format(idx)
                import IPython as IP
                IP.embed()  # noqa
            else:
                indices_set.add(idx)

    print len(indices), len(indices_set)

    import IPython as IP
    IP.embed()  # noqa
    # for i in xrange(2):
    #     TestDataSpeed(ds, size=100, warmup=10).start()
