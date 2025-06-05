from PIL import Image,ImageFilter
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10

class GaussianBlur(object):  
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):  
        self.kernel_size = kernel_size  
        self.sigma = sigma  
          
    def __call__(self, img):  
        sigma = random.uniform(self.sigma[0], self.sigma[1])  
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))  
        return img  


class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

# 预训练模式下，对每个图像生成两个不同的增强视图用于对比学习；而在微调模式下，只对每个图像应用较少的增强
train_transform = transforms.Compose([  
    transforms.RandomResizedCrop(32),  
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  
    transforms.RandomGrayscale(p=0.2),  
    transforms.RandomApply([GaussianBlur(kernel_size=int(32 * 0.1) | 1)], p=0.5),  # 添加高斯模糊  
    transforms.ToTensor(),  
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  
])  
  
test_transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  
])

import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



# 在助教的代码文件里只提供了裁剪，翻转，
# 颜色抖动，灰度图这几个数据增强操作，我参考了SimCLR的源码进行了一些附加操作
# 高斯模糊特别重要，因为它能够帮助模型关注图像的全局结构而不是局部细节。在原始论文中，作者发现移除高斯模糊会导致性能下降。
'''
def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
  """Blurs the given image with separable convolution.


  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be


  """
  del width
  def _transform(image):
    sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
    return gaussian_blur(
        image, kernel_size=height//10, sigma=sigma, padding='SAME')
  return random_apply(_transform, p=p, x=image)
'''