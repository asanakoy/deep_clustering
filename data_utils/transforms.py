import numpy as np
from scipy import ndimage
from PIL import Image
import torchvision.transforms as transforms

class Sobel(object):
    """Apply Sobel filter given PIL Image.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be filtered.

        Returns:
            PIL Image: Gradient magnitudes of the image.
        """
        img = np.asarray(img)
        dx = ndimage.sobel(img, 1)
        dy = ndimage.sobel(img, 0)
        mag = np.hypot(dx, dy)  # magnitude
        mag *= 255.0 / mag.max()  # normalize
        return Image.fromarray(mag)


IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])