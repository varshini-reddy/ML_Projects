import zipfile
import os
import sys
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

from PIL import Image
import detectron2 
from detectron2.utils.logger import setup_logger
setup_logger()
from google.colab.patches import cv2_imshow

from IPython.display import clear_output
# import some common libraries
import sys
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
from scipy.ndimage.filters import gaussian_filter
import importlib

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
np.set_printoptions(threshold=sys.maxsize)

# Detic libraries
sys.path.insert(0, '/content/Detic/third_party/CenterNet2')
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test
import matplotlib.pyplot as plt

import os
import cv2

import models as md
from tqdm import tqdm
from glob import glob
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
from google.colab.patches import cv2_imshow
from scipy.ndimage.filters import gaussian_filter


import matplotlib.pyplot as plt
from torch.autograd import Variable
device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")



def get_inpaint_model():
  net = importlib.import_module("in_painting.src.model.aotgan")
  inpaint_model = net.InpaintGenerator([1,2,5,8], 8)
  inpaint_model.load_state_dict(torch.load("/content/in_painting/experiments/G0000000.pt", map_location='cpu'))
  return inpaint_model


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image



def inpaint(model, image, mask1):

    image_masked = (image * (1 - mask1).float()) + mask1
    model.eval()

    with torch.no_grad():
      pred_img = model(image_masked, mask1 )
    
    # Using altered mask after altered prediction
    try:
      comp_imgs = (1 - mask1) * image + mask1 * pred_img
    except:
      w,h = mask1.shape[2:]

      pred_img = T.Resize((w,h))(pred_img)
      comp_imgs = (1 - mask1) * image + mask1 * pred_img
    comp_np = postprocess(comp_imgs[0])
    return comp_np

