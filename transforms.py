import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from PIL import Image

# bbox为bounding box，即框住对象物体的框（的两个坐标）
# 大部分F函数的image同时支持PIL格式和torch.tensorge格式


def pad_if_smaller(img, size, fill=0):
    # 将图片（torch.Tenser)填充至size大小的正方形
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, padw, padh], fill=fill)  # 由于报错从tuple改成了list（检修时优先注意此处）
    return img


class Compose(object):
    # 使用transforms数组（？）处理图像（transform来源待补充）
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox):
        for t in self.transforms:
            image, bbox = t(image, bbox)
        return image, bbox


class LoadImage(object):
    # image格式为file（文件路径），返回PIL Image类型
    def __call__(self, image, bbox):
        image = Image.open(image)
        return image, bbox


class Resize(object):
    # 更改分辨率，一般为降
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize((size, size))  # 更改分辨率，一般为降

    def __call__(self, image, bbox):
        bbox = [b * self.size / image.size[-1] for b in bbox]
        image = self.resize(image)
        return image, bbox


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, bbox):
        w, h = image.size[-2:]
        ratio = random.uniform(self.min_size, self.max_size)  # 返回minsize，maxsize之间的两个随机浮点数
        hp = int(h * ratio)
        wp = int(w * ratio)
        size = (hp, wp)
        image = F.resize(image, size)  # 将图片随机降一定的分辨率
        bbox = [b * hp / h for b in bbox]  #bbox相应降低
        return image, bbox


class RandomHorizontalFlip(object):
    # 随机水平翻转
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, bbox):
        if random.random() < self.flip_prob:  # 生成随机[0，1)浮点数
            image = F.hflip(image)
            bbox = [
                image.size[-1] - bbox[2], bbox[1], image.size[-1] - bbox[0],
                bbox[2]
            ]
        return image, bbox


class RandomCrop(object):
    # 随机选择图片的一个部分
    def __init__(self, size):
        self.size = size

    def __call__(self, image, bbox):
        image = pad_if_smaller(image, self.size[0])
        crop_params = transforms.RandomCrop.get_params(image, self.size)
        # PIL Image or Tensor
        image = F.crop(image, *crop_params)
        bbox = [
            bbox[0] - crop_params[0], bbox[1] - crop_params[1],
            bbox[2] - crop_params[0], bbox[3] - crop_params[1]
        ]
        return image, bbox

# class RandomColor(object):
#     # 随机选择图片的一个部分
#     def __init__(self):
#         self.size = size
#
#     def __call__(self, image, bbox):
#         #image = pad_if_smaller(image, self.size[0])
#         crop_params = transforms.RandomCrop.get_params(image, self.size)
#         # PIL Image or Tensor
#         image = F.crop(image, *crop_params)
#         return image, bbox
#tr.ColorJitter(brightness, contrast, saturation, hue)

class ToTensor(object):
    # Convert a PIL Image or numpy.ndarray to tensor.
    def __call__(self, image, bbox):
        image = F.to_tensor(image)
        bbox = torch.Tensor(bbox)
        return image, bbox


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, bbox):
        image = F.normalize(image, mean=self.mean, std=self.std)
        bbox
        return image, bbox

# https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py包含本库的大量函数