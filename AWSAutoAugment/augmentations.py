# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose

random_mirror = True


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v): 
    v = int(v)
    return PIL.ImageOps.posterize(img, v)



def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def augment_list():  # 36 operations
    l = [
        (ShearX, -0.1, 0.1),  # 0
        (ShearX, -0.2, 0.2),  # 1     
        (ShearX, -0.3, 0.3),  # 2    
        (ShearY, -0.1, 0.1),  # 3
        (ShearY, -0.2, 0.2),  # 4        
        (ShearY, -0.3, 0.3),  # 5
        (TranslateX, -0.15, 0.15),  # 6
        (TranslateX, -0.3, 0.3),    # 7
        (TranslateX, -0.45, 0.45),  # 8
        (TranslateY, -0.15, 0.15),  # 9
        (TranslateY, -0.3, 0.3),    # 10      
        (TranslateY, -0.45, 0.45),  # 11
        (Rotate, -10, 10),  # 12
        (Rotate, -20, 20),  # 13
        (Rotate, -30, 30),  # 14 
        (Color, 0.1, 0.3),  # 15 
        (Color, 0.1, 0.6),  # 16         
        (Color, 0.1, 0.9),  # 17       
        (Solarize, 0, 26),  # 18
        (Solarize, 0, 102),  # 19
        (Solarize, 0, 179),  # 20       
        (Posterize, 4, 4.4),  # 21
        (Posterize, 4, 5.6),  # 22    
        (Posterize, 4, 6.8),  # 23        
        (Contrast, 0.1, 1.3),  # 24   
        (Contrast, 0.1, 1.6),  # 25     
        (Contrast, 0.1, 1.9),  # 26
        (Sharpness, 0.1, 1.3),  # 27
        (Sharpness, 0.1, 1.6),  # 28
        (Sharpness, 0.1, 1.9),  # 29        
        (Brightness, 0.1, 1.9),  # 30       
        (Brightness, 0.1, 1.9),  # 31     
        (Brightness, 0.1, 1.9),  # 32             
        (AutoContrast, 0, 1),  # 33
        (Equalize, 0, 1),  # 34          
        (Invert, 0, 1),  # 35
        ]
    return l

def augment_list_by_name():  # 36 operations
    l = [
        ('ShearX',0.1),  # 0
        ('ShearX', 0.2),  # 1     
        ('ShearX', 0.3),  # 2    
        ('ShearY', 0.1),  # 3
        ('ShearY', 0.2),  # 4        
        ('ShearY', 0.3),  # 5
        ('TranslateX', 0.15),  # 6
        ('TranslateX', 0.3),    # 7
        ('TranslateX', 0.45),  # 8
        ('TranslateY', 0.15),  # 9
        ('TranslateY', 0.3),    # 10      
        ('TranslateY', 0.45),  # 11
        ('Rotate', 10),  # 12
        ('Rotate', 20),  # 13
        ('Rotate', 30),  # 14 
        ('Color', 0.3),  # 15 
        ('Color', 0.6),  # 16         
        ('Color', 0.9),  # 17       
        ('Solarize', 26),  # 18
        ('Solarize', 102),  # 19
        ('Solarize', 179),  # 20       
        ('Posterize', 4.4),  # 21
        ('Posterize', 5.6),  # 22    
        ('Posterize', 6.8),  # 23        
        ('Contrast', 1.3),  # 24   
        ('Contrast', 1.6),  # 25     
        ('Contrast', 1.9),  # 26
        ('Sharpness', 1.3),  # 27
        ('Sharpness', 1.6),  # 28
        ('Sharpness', 1.9),  # 29        
        ('Brightness', 1.9),  # 30       
        ('Brightness', 1.9),  # 31     
        ('Brightness', 1.9),  # 32             
        ('AutoContrast', 1),  # 33
        ('Equalize', 1),  # 34          
        ('Invert', 1),  # 35
        ]
    return l

augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

class UniformAugmentation(object):
    def __init__(self):
        self.aug_list = augment_list()

    def __call__(self, img):
        augment_fn, low, high = random.choice(self.aug_list)
        img = augment_fn(img.copy(), high)
        return img
    
class AWSAugmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for augment_fn, low, high in policy:
                img = augment_fn(img.copy(), high)
        return img