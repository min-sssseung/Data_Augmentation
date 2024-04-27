from PIL import Image, ImageEnhance, ImageOps
import random

# 논문에서 magnitude에 대해 10가지 값으로 진행
# We discretize the range of magnitudes into 10 values (uniform spacing) 라고 명시
# 구현에서는 -1, 1 고르고 1~10 을 곱하는 식으로 진행


# Image.Affine -> Image.AFFINE(a,b,tx, c,d,ty)
# (a, b) 및 (c, d)는 각각 X 및 Y 축에서의 이미지의 크기 조절을 의미, tx,ty는 각 축으로의 이동을 의미한다.



# SHEAR X,Y :  Shear the image along the horizontal (vertical) axis with rate magnitude
# Image.BICUBIC : image interpolation 기법 -> 16개의 주변 픽셀을 활용하여 가중치를 계산한 후 새로운 픽셀에 도입

class ShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)

class ShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class TranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor)


class TranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor)

# x.convert('RGBA') -> r,g,b,a(투명)으로 변경 후 rotate
# 이후 투명한 rotate전 사이즈에 rotate시킨 이미지를 합성한다.

class Rotate(object):
    # from https://stackoverflow.com/questions/
    # 5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)


# [ImageEnhance]
# Color.enhance -> 0: 흑백, 1: 원본 , 1이 넘어가면 색상이 더해진다.
# Contrast ->  0 에 가까울 수록 대비가 없는 회색 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 대비가 강해지는 결과
# Brightness -> 0 에 가까울 수록 그냥 검정 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 밝기가 강해지는 결과
# Sharpness ->  0 에 가까울 수록 이미지는 흐릿한 이미지에 가깝게 되고 1 이 원본 값이고 1이 넘어가면 원본에 비해 선명도가 강해지는 결과
 

class Color(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))

class Contrast(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Sharpness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Brightness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))


# [ImageOps]
# Invert -> 색상 반전
# Posterize -> 색상 제한 magnitude만큼의 색상으로 제한됨
# Solarize -> magnitude이상(threshold)의 모든 픽셀값을 반전
# Autocontrast -> 이미지 contrast 자동으로 설정
# Equalize -> This function applies a non-linear mapping to the input image, 
# in order to create a uniform distribution of grayscale values in the output image


class Posterize(object):
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class Solarize(object):
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class AutoContrast(object):
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class Equalize(object):
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)

class Invert(object):
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)
