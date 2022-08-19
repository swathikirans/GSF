import torchvision
import random
from PIL import Image, ImageOps, ImageEnhance, ImageChops
import PIL
import numpy as np
import numbers
import math
import torch
import re
# from skimage.transform import resize, rotate


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):

        img_group, label = img

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images, label


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img):
        img_group, label = img
        return [self.worker(img) for img in img_group], label


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False, target_transform=None):
        self.is_flow = is_flow
        self.target_transform = target_transform

    def __call__(self, img, is_flow=False):
        img_group, label = img
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            if self.target_transform is not None:
                if label in self.target_transform:
                    label = self.target_transform[label]
            return ret, label
        else:
            return img_group, label


class AutoAugment(object):
    def __init__(self):
        self.augmentor = torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.IMAGENET)
    
    def __call__(self, tensor_label):
        tensor, label = tensor_label
        tensor = self.augmentor(tensor.view((-1, 3) + tensor.size()[-2:])).view((-1)+ tensor.size()[-2:])
        return tensor, label

class GroupNormalize(object):
    def __init__(self, mean, std, div=True):
        self.mean = mean
        self.std = std
        self.div = div

    def __call__(self, tensor_label):
        tensor, label = tensor_label
        tensor = tensor.float()
        if self.div:
            tensor = tensor / 255.0
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor, label


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img):
        img_group, label = img
        return [self.worker(img) for img in img_group], label


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [rotate(image=img, angle=angle, preserve_range=True) for img in clip]
        elif isinstance(clip[0], Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img):
        img_group, label = img
        if self.scale_worker is not None:
            img_group, _ = self.scale_worker((img_group, label))

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group, label

class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img):
        img_group, label = img

        if self.scale_worker is not None:
            img_group, _ = self.scale_worker((img_group, label))

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group, label

class GroupFiveCrops(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img):
        img_group, label = img
        if self.scale_worker is not None:
            img_group, _ = self.scale_worker((img_group, label))

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)

            oversample_group.extend(normal_group)
        return oversample_group, label


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):

        img_group, label = img
        im_size = img_group[0].size
        # print('size = ', im_size)

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group, label

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img_group, label = img
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group)), label


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img):
        img_group, label = img
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label
            else:
                return np.concatenate(img_group, axis=2), label


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    
    def __call__(self, pic_label):
        pic, label = pic_label
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img, label


class IdentityTransform(object):

    def __call__(self, data):
        return data

"""randaugment code from https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
"""

_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

_LEVEL_DENOM = 10.  # denominator for conversion from 'Mx' magnitude scale to fractional aug level for op arguments

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs['resample'])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _LEVEL_DENOM) * 30.
    level = _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return (level / _LEVEL_DENOM) * 1.8 + 0.1,


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9] if level <= _LEVEL_DENOM
    level = (level / _LEVEL_DENOM) * .9
    level = max(0.1, 1.0 + _randomly_negate(level))  # keep it >= 0.1
    return level,


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _LEVEL_DENOM) * 0.3
    level = _randomly_negate(level)
    return level,


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _LEVEL_DENOM) * float(translate_const)
    level = _randomly_negate(level)
    return level,


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get('translate_pct', 0.45)
    level = (level / _LEVEL_DENOM) * translate_pct
    level = _randomly_negate(level)
    return level,


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 4),


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return 4 - _posterize_level_to_arg(level, hparams)[0],


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 4) + 4,


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 256),


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return 256 - _solarize_level_to_arg(level, _hparams)[0],


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return int((level / _LEVEL_DENOM) * 110),

NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'PosterizeIncreasing': posterize,
    'PosterizeOriginal': posterize,
    'Solarize': solarize,
    'SolarizeIncreasing': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
}

LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Rotate': _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    'Posterize': _posterize_level_to_arg,
    'PosterizeIncreasing': _posterize_increasing_level_to_arg,
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeIncreasing': _solarize_increasing_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'ColorIncreasing': _enhance_increasing_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'ContrastIncreasing': _enhance_increasing_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'BrightnessIncreasing': _enhance_increasing_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'SharpnessIncreasing': _enhance_increasing_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
}

class AugmentOp:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution
        self.magnitude_std = self.hparams.get('magnitude_std', 0)
        self.magnitude_max = self.hparams.get('magnitude_max', None)

    def __call__(self, imgs):
        if self.prob < 1.0 and random.random() > self.prob:
            return imgs
        magnitude = self.magnitude
        if self.magnitude_std > 0:
            # magnitude randomization enabled
            if self.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        # default upper_bound for the timm RA impl is _LEVEL_DENOM (10)
        # setting magnitude_max overrides this to allow M > 10 (behaviour closer to Google TF RA impl)
        upper_bound = self.magnitude_max or _LEVEL_DENOM
        magnitude = max(0., min(magnitude, upper_bound))
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return [self.aug_fn(img, *level_args, **self.kwargs) for img in imgs[0]], imgs[1]

    def __repr__(self):
        fs = self.__class__.__name__ + f'(name={self.name}, p={self.prob}'
        fs += f', m={self.magnitude}, mstd={self.magnitude_std}'
        if self.magnitude_max is not None:
            fs += f', mmax={self.magnitude_max}'
        fs += ')'
        return fs


_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    #'Cutout'  # NOTE I've implement this as random erasing separately
]


_RAND_INCREASING_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'SolarizeAdd',
    'ColorIncreasing',
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    #'Cutout'  # NOTE I've implement this as random erasing separately
]



# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    'Rotate': 0.3,
    'ShearX': 0.2,
    'ShearY': 0.2,
    'TranslateXRel': 0.1,
    'TranslateYRel': 0.1,
    'Color': .025,
    'Sharpness': 0.025,
    'AutoContrast': 0.025,
    'Solarize': .005,
    'SolarizeAdd': .005,
    'Contrast': .005,
    'Brightness': .005,
    'Equalize': .005,
    'Posterize': 0,
    'Invert': 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AugmentOp(
        name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs


def rand_augment_transform(config_str, hparams):
    """
    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied, or uniform sampling if infinity (or > 100)
        'mmax' - set upper bound for magnitude to something other than default of  _LEVEL_DENOM (10)
        'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    magnitude = _LEVEL_DENOM  # default to _LEVEL_DENOM for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param / randomization of magnitude values
            mstd = float(val)
            if mstd > 100:
                # use uniform sampling in 0 to magnitude if mstd is > 100
                mstd = float('inf')
            hparams.setdefault('magnitude_std', mstd)
        elif key == 'mmax':
            # clip magnitude between [0, mmax] instead of default [0, _LEVEL_DENOM]
            hparams.setdefault('magnitude_max', int(val))
        elif key == 'inc':
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'
    ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


if __name__ == "__main__":
    trans = torchvision.transforms.Compose([
        GroupScale(256),
        GroupRandomCrop(224),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )]
    )

    im = Image.open('../tensorflow-model-zoo.torch/lena_299.png')

    color_group = [im] * 3
    rst = trans(color_group)

    gray_group = [im.convert('L')] * 9
    gray_rst = trans(gray_group)

    trans2 = torchvision.transforms.Compose([
        GroupRandomSizedCrop(256),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225])
    ])
    print(trans2(color_group))


