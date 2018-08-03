import json
import torchvision.datasets
from torchvision.datasets.folder import find_classes
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import IMG_EXTENSIONS


def index_imagenet(root, index_save_path):
    """
    Index a data folder, where samples are arranged in this way:

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        index_save_path (string): Where to save index.

    Returns:
        index (dict) containing the following items:
            classes (list): List of the class names.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
    """
    classes, class_to_idx = find_classes(root)
    samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
    if len(samples) == 0:
        raise (RuntimeError("Found 0 files in subfolders of: " + root +
                            "\nSupported extensions are: " + ",".join(IMG_EXTENSIONS)))
    index = {
        'classes': classes,
        'class_to_idx': class_to_idx,
        'samples': samples
    }
    with open(index_save_path, 'w') as json_file:
        json.dump(index, json_file, indent=0)
    return index


class IndexedDataset(torchvision.datasets.ImageFolder):
    """A data loader initialized from the indexed folder

    Is useful, because traversing all directories of a large dataset takes a lot of time.

    Args:
        root (string): Root directory path.
        index (dict): a dict containing classes, class_to_idx and samples.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    # noinspection PyMissingConstructor
    def __init__(self, root, index,
                 loader=default_loader, transform=None, target_transform=None):
        if len(index['samples']) == 0:
            raise (RuntimeError("Empty index: " + root))

        self.root = root
        self.loader = loader

        self.classes = index['classes']
        self.class_to_idx = index['class_to_idx']
        self.samples = index['samples']

        self.transform = transform
        self.target_transform = target_transform
