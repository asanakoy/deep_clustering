import numpy as np
import cv2
import os

from tensorpack.dataflow import *
from tensorpack.dataflow.serialize import LMDBSerializer


class Dummy(DataFlow):

    def get_data(self):

        for cur_id in xrange(1000000):
            yield [cur_id]

    def size(self):
        return 1000000


if __name__ == '__main__':
    output = os.path.expanduser('~/workspace/tmp/dummy_1M.lmdb')
    LMDBSerializer.save(Dummy(), output, write_frequency=100000)
