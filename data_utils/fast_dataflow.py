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

_use_shared_memory = False


class TorchBatchData(BatchData):
    """
    Stack datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The batch can be either a list of original components, or (by default)
    a Tensor of original components.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guaranteed to have the same batch size.
                If set to True, `ds.size()` must be accurate.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of a Tensor with extra dimension.
        """
        super(TorchBatchData, self).__init__(ds, batch_size, remainder, use_list)
        # for compatibility with torch DataLoader
        self.dataset = ds

    def __iter__(self):
        self.reset_state()
        return self.get_data()

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False):
        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"

        size = len(data_holder[0])
        result = []
        for k in range(size):
            if use_list:
                result.append(
                    [x[k] for x in data_holder])
            else:
                dt = data_holder[0][k]
                batch = [x[k] for x in data_holder]
                if type(dt) in list(six.integer_types) + [bool]:
                    tp = 'int32'
                elif type(dt) == float:
                    tp = 'float32'
                else:
                    try:
                        tp = dt.dtype
                    except AttributeError:
                        raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                try:
                    if isinstance(dt, torch.Tensor):
                        out = None
                        if _use_shared_memory:
                            # If we're in a background process, concatenate directly into a
                            # shared memory tensor to avoid an extra copy
                            numel = sum([b.numel() for b in batch])
                            storage = data_holder[0][k].storage()._new_shared(numel)
                            out = data_holder[0][k].new(storage)
                        result.append(torch.stack(batch, 0, out=out))
                    elif type(dt).__name__ == 'ndarray':
                        # array of string classes and object
                        if re.search('[SaUO]', dt.dtype.str) is not None:
                            raise TypeError(error_msg.format(dt.dtype))
                        result.append(torch.stack([torch.from_numpy(b) for b in batch], 0))
                    elif isinstance(dt, six.integer_types):
                        result.append(torch.LongTensor(batch))
                    elif isinstance(dt, float):
                        result.append(torch.DoubleTensor(batch))
                    elif isinstance(dt, six.string_types):
                        result.append(batch)
                    else:
                        raise TypeError((error_msg.format(type(dt))))
                except Exception as e:  # noqa
                    logger.exception(e.message)
                    logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                    if isinstance(dt, np.ndarray):
                        s = pprint.pformat([x[k].shape for x in data_holder])
                        logger.error("Shape of all arrays to be batched: " + s)
                    try:
                        # open an ipython shell if possible
                        import IPython as IP
                        IP.embed()  # noqa
                    except ImportError:
                        pass
        return result


class TorchAugmentorList(AugmentorList):
    """
    Augment by a torchvision transform
    """

    # noinspection PyMissingConstructor
    def __init__(self, transform):
        """
        Args:
            augmentors (list): list of :class:`ImageAugmentor` instance to be applied.
        """
        self.transform = transform

    def augment(self, d):
        """
        Perform augmentation on the data.
        """
        return self.transform(d)

    def reset_state(self):
        pass


def create_lmdb_stream(lmdb_path, shuffle=False, new_labels=None, return_index=False):
    from tensorpack.utils.serialize import loads
    ds = LMDBData(lmdb_path, shuffle=shuffle)
    if shuffle:
        logger.warning('Performance may suffer when loading LMDB with shuffle=True.')

    def desearialize_data_point(dp):
        idx = int(dp[0])  # index from 0 to N
        value = loads(dp[1])
        assert len(value) == 2, len(value)
        img_encoded, label = value
        if new_labels is not None:
            label = new_labels[idx]
        if return_index:
            return img_encoded, label, idx
        else:
            return img_encoded, label

    return MapData(ds, desearialize_data_point)


def create_fast_lmdb_flow(lmdb_path, nr_proc=10, batch_size=256, shuffle=False, return_index=False):
    # image_index = 0
    # ds = LMDBSerializer.load(lmdb_path, shuffle=shuffle)
    ds = create_lmdb_stream(lmdb_path, shuffle=shuffle, return_index=return_index)
    # ds = LocallyShuffleData(ds, buffer_size=10000)
    # We use PrefetchData to launch the base LMDB Flow in only one process,
    # and only parallelize the transformations with another
    ds = PrefetchData(ds, nr_prefetch=5000, nr_proc=1)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), index=0)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        pil_to_np_array,
        IMAGENET_NORMALIZE_NP
    ])
    ds = AugmentImageComponent(ds, TorchAugmentorList(transform), index=0, copy=False)
    ds = PrefetchDataZMQ(ds, nr_proc=nr_proc)
    ds = TorchBatchData(ds, batch_size=batch_size, remainder=True)
    return ds


# TODO: add __len__ method: __len__ = size
# DataLoader.len() = number of batches

if __name__ == '__main__':
    split ='train'
    lmdb_path = '/export/home/asanakoy/workspace/datasets/ILSVRC2012/{}.lmdb'.format(split)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('-j', '--njobs', type=int, default=10)
    args = parser.parse_args()
    ds = create_fast_lmdb_flow(lmdb_path, nr_proc=args.njobs, batch_size=256, shuffle=args.shuffle, return_index=True)
    print 'len(ds):', len(ds)

    indices = []
    indices_set = set()
    for img, label, idxb in tqdm(ds):
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
