from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter
import numpy as np
import random
import torch

PARAMETER_MAX = 10


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def contrast(img, factor=2.0):
    return ImageEnhance.Contrast(img).enhance(factor)


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def sharpness(img, factor=3.0):
    return ImageEnhance.Sharpness(img).enhance(factor)


def brightness(img, factor=0.5):
    return ImageEnhance.Brightness(img).enhance(factor)


def Identity(img):
    return img


def invert(img, factor):
    if factor:
        return ImageOps.invert(img)
    else:
        return img


def equalize(img, factor):
    if factor:
        return ImageOps.equalize(img)
    else:
        return img


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, factor, bias=0, threshold=128):
    v, max_v = factor
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return ImageOps.solarize(img, threshold)


def gauss_noise(image, params):
    row, col, ch = image.shape
    mean, var = params
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


def Cutout(img, v, fill=0, bias=0):
    if v == 0:
        return img

    v = int(v * min(img.size))
    return CutoutAbs(img, v, fill)


def CutoutAbs(img, v, fill, **kwarg):
    # w, h = img.size
    # x0 = np.random.uniform(0, w)
    # y0 = np.random.uniform(0, h)
    x0 = np.random.uniform(128, 384)
    y0 = np.random.uniform(64, 256)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, fill=fill, outline=fill)
    return img


def Cutout1(img, coord, color=0):
    img = img.copy()
    ImageDraw.Draw(img).rectangle(coord, fill=color, outline=color)
    return img


def TranslateX(img, v):
    # v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    # v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):
    # v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def mixup_one_target(x, y, alpha=1.0, device='cuda', is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam


def Posterize(img, v):
    # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return ImageOps.posterize(img, v)


def Edge_enhance(img, v):
    if v:
        return img.filter(ImageFilter.EDGE_ENHANCE)
    else:
        return img


def Edge_enhance_more(img, v):
    if v:
        return img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    else:
        return img


def Blur(img, v):
    if v:
        return img.filter(ImageFilter.BLUR)
    else:
        return img


def Detail(img, v):
    if v:
        return img.filter(ImageFilter.DETAIL)
    else:
        return img


def Smooth(img, v):
    if v:
        return img.filter(ImageFilter.SMOOTH)
    else:
        return img


def Smooth_more(img, v):
    if v:
        return img.filter(ImageFilter.SMOOTH_MORE)
    else:
        return img


def Sharpen(img, v):
    if v:
        return img.filter(ImageFilter.SHARPEN)
    else:
        return img


def Contour(img, v):
    if v:
        return img.filter(ImageFilter.CONTOUR)
    else:
        return img


def Emboss(img, v):
    if v:
        return img.filter(ImageFilter.EMBOSS)
    else:
        return img


def Find_edges(img, v):
    if v:
        return img.filter(ImageFilter.FIND_EDGES)
    else:
        return img


def AutoContrast(img, v):
    return ImageOps.autocontrast(img, v)


def mixup(x, y, device, alpha=0.75):
    x1, x2 = x
    y1, y2 = y
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    index = torch.randperm(x1.size(0)).to(device)
    # x1 = x1[index]; x2 = x2[index];
    mixed_x = lam * x1 + (1 - lam) * x2[index]
    mixed_y = lam * y1 + (1 - lam) * y2[index]
    return mixed_x.detach(), mixed_y.detach()


def dataAugment(img, da, factor=0):
    if da == "contrast":
        img = contrast(img, factor)
    elif da == "sharpness":
        img = sharpness(img, factor)
    elif da == "brightness":
        img = brightness(img, factor)
    elif da == "gaussian_noise":
        img = gauss_noise(img, factor)
    elif da == "invert":
        img = invert(img, factor)
    elif da == "equalize":
        img = equalize(img, factor)
    elif da == "solarize":
        img = SolarizeAdd(img, factor)
    elif da == "identity":
        img = Identity(img)
    elif da == "posterize":
        img = Posterize(img, factor)
    elif da == "blur":
        img = Blur(img, factor)
    elif da == "smooth":
        img = Smooth(img, factor)
    elif da == "edge_enhance":
        img = Edge_enhance(img, factor)
    elif da == "edge_enhance_more":
        img = Edge_enhance_more(img, factor)
    elif da == "detail":
        img = Detail(img, factor)
    elif da == "emboss":
        img = Emboss(img, factor)
    elif da == "find_edges":
        img = Find_edges(img, factor)
    elif da == "smooth_more":
        img = Smooth_more(img, factor)
    elif da == "sharpen":
        img = Sharpen(img, factor)
    elif da == "contour":
        img = Contour(img, factor)
    elif da == "autocontrast":
        img = AutoContrast(img, factor)
    else:
        raise Exception("data augmentation method not found.")
    return img


def factor(da):
    if da == "contrast":
        factor1, factor2 = random.sample(
            [round(num, 1) for num in np.arange(0.3, 1.4, 0.1) if not round(num, 1) == 1.0], 2)

    elif da == "brightness":
        factor1, factor2 = random.sample(
            [round(num, 1) for num in np.arange(0.3, 1.4, 0.1) if not round(num, 1) == 1.0], 2)

    elif da == "sharpness":
        factor1, factor2 = random.sample(
            [round(num, 1) for num in np.arange(0.1, 2.0, 0.1) if not round(num, 1) == 1.0], 2)

    elif da == "autocontrast":
        factor1, factor2 = random.sample(
            [round(num, 2) for num in np.arange(0.01, 0.21, 0.01) if not round(num, 2) == 1.0], 2)

    elif da == "invert":
        factor1 = 1
        factor2 = 0

    elif da == "equalize":
        factor1 = 1
        factor2 = 0

    elif da == "blur":
        factor1 = 1
        factor2 = 0

    elif da == "smooth":
        factor1 = 1
        factor2 = 0

    elif da == "edge_enhance":
        factor1 = 1
        factor2 = 0

    elif da == "edge_enhance_more":
        factor1 = 1
        factor2 = 0

    elif da == "detail":
        factor1 = 1
        factor2 = 0

    elif da == "emboss":
        factor1 = 1
        factor2 = 0

    elif da == "find_edges":
        factor1 = 1
        factor2 = 0

    elif da == "smooth_more":
        factor1 = 1
        factor2 = 0

    elif da == "contour":
        factor1 = 1
        factor2 = 0

    elif da == "sharpen":
        factor1 = 1
        factor2 = 0

    elif da == "posterize":
        factor1, factor2 = random.sample([4, 5, 6, 7, 8], 2)

    elif da == "solarize":
        factor1_1, factor2_1 = random.sample([1, 2, 3], 2)
        factor1 = (256, factor1_1)
        factor2 = (256, factor2_1)

    elif da == "gaussian_noise":
        var = 0.01
        mean = 0.0
        factor1 = (var, mean)
        factor2 = (var, mean)

    elif da == "identity":
        factor1 = 0
        factor2 = 0

    else:
        raise Exception("data augmentation method not found.")

    return factor1, factor2
