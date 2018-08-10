import numpy as np
from scipy import ndimage
from PIL import Image
import torch
import torchvision.transforms as transforms
import warnings

# class Sobel(object):
#     """Apply Sobel filter given PIL Image.
#     """
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): Image to be filtered.
#
#         Returns:
#             PIL Image: Gradient magnitudes of the image.
#         """
#         img = np.asarray(img)
#         assert len(img.shape) == 2
#         dx = ndimage.sobel(img, 1)
#         dy = ndimage.sobel(img, 0)
#         mag = np.hypot(dx, dy)  # magnitude
#         mag /= mag.max()  # normalize in range [0, 1]
#         return Image.fromarray(mag, mode='F')


class NormalizeNp(object):
    """Normalize numpy image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``np.ndarray`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        assert len(self.mean) == 3
        assert len(self.std) == 3

    def __call__(self, array):
        """
        Args:
            array (np.ndarray): Tensor image of size (C, H, W) to be normalized.
            Modifies the array inplace.
        Returns:
            Tensor: Normalized Tensor image.
        """
        assert array.shape[0] == 3, array.shape
        for i in xrange(array.shape[0]):
            array[i, ...] -= self.mean[i]
            array[i, ...] /= self.std[i]
        return array

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def pil_to_np_array(pic):
    """Convert a ``PIL Image`` to ``numpy.ndarray``.
       1. If image is 1-channel -> replicate to make 3 channels.
       2. Swap dimensions HWC -> CHW,
       3. BGR -> RGB
       3. Normalize array to range [0., 1.] if possible (i.e when image is of np.uint8 or np.bool type).
    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        numpy.ndarray: Converted image of types from {np.float32, np.int16, np.int32}
    """
    if not isinstance(pic, Image.Image):
        raise TypeError('pic should be PIL Image Got {}'.format(type(pic)))

    # handle PIL Image
    if pic.mode == 'I':
        # (32-bit signed integer pixels) unscaled
        # Represented as int32
        img = np.array(pic, np.int32, copy=False)
    elif pic.mode == 'I;16':
        # (16-bit signed integer pixels) unscaled
        img = np.array(pic, np.int16, copy=False)
    elif pic.mode == 'F':
        # (32-bit floating point pixels) unscaled
        img = np.array(pic, np.float32, copy=False)
    elif pic.mode == '1':
        # (1-bit pixels, black and white, stored with one pixel per byte).
        # Represented as bool
        img = 255 * np.array(pic, np.uint8, copy=False)
    else:
        # L: (8-bit pixels, black and white)
        # P: (8-bit pixels, mapped to any other mode using a color palette)
        # RGB, RGBA, CMYK, YCbCr, LAB, HSV
        img = np.array(pic, np.uint8, copy=False)
    if pic.mode in ['I', 'I;16', 'F']:
        warnings.warn('Strange image type: I;16. It will not be normalized to [0., 1.] range.')

    if len(img.shape) == 2:
        img = np.repeat(img[..., np.newaxis], 3, axis=2)

    # put it from HWC to CHW format and BGR -> RGB
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose((2, 0, 1))[::-1, ...]
    # img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if img.dtype == np.uint8:
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    else:
        img = np.ascontiguousarray(img)
    return img


SIMPLE_NORMALIZE = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])

# Assuming the channels are stored in RGB order
IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
# Assuming the channels are stored in RGB order
IMAGENET_NORMALIZE_NP = NormalizeNp(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
