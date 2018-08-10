#!/bin/bash
python imagenet_utils/extract_lmdb_index_label.py --data ~/workspace/datasets/ILSVRC2012 --name train --output ~/workspace/datasets/ILSVRC2012/train_lmdb_index.npy
python imagenet_utils/extract_lmdb_index_label.py --data ~/workspace/datasets/ILSVRC2012 --name val --output ~/workspace/datasets/ILSVRC2012/val_lmdb_index.npy
