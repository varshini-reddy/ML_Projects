import cv2 
import numpy as np
import torch as tr
import torch.nn as nn
from PIL import Image
from torch.nn.functional import interpolate
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.stats import percentileofscore
import numpy as np
from torch.utils import data
from PIL import Image


def alter_mask(mask, val = 50):
	inp_mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (val, val)))
    inp_mask = cv2.dilate(inp_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (val, val)))
    return inp_mask


def ImageTransform(path, ip_type="filename"):
  if ip_type=="filename":
    file_name = path
    img = Image.open(file_name)
    img = img.convert("RGB")
  else:
    img = path

  transform1 = T.Compose([T.ToTensor()])

  x = transform1(img)*255

  img_mean = tr.Tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
  img_std = tr.Tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

  x = x.to(dtype=tr.float)

  x = interpolate(
      x.unsqueeze(0),
      size=(456,456),#(model.img_size, model.img_size),
      mode='bilinear',
      align_corners=False,
      recompute_scale_factor=False
  ).squeeze(0)

  x = ((x - img_mean) / img_std)#.unsqueeze(0)

  # x = x.to(device)

  return np.array(img), x
