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


def get_predictor():
	cfg = get_cfg()
	add_centernet_config(cfg)
	add_detic_config(cfg)
	cfg.merge_from_file("/content/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
	cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
	cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
	# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
	predictor = DefaultPredictor(cfg)

	# Setup the model's vocabulary using build-in datasets

	BUILDIN_CLASSIFIER = {
	    'lvis': 'Detic/datasets/metadata/lvis_v1_clip_a+cname.npy',
	    'objects365': 'Detic/datasets/metadata/o365_clip_a+cnamefix.npy',
	    'openimages': 'Detic/datasets/metadata/oid_clip_a+cname.npy',
	    'coco': 'Detic/datasets/metadata/coco_clip_a+cname.npy',
	}

	BUILDIN_METADATA_PATH = {
	    'lvis': 'lvis_v1_val',
	    'objects365': 'objects365_v2_val',
	    'openimages': 'oid_val_expanded',
	    'coco': 'coco_2017_val',
	}

	vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
	metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
	classifier = BUILDIN_CLASSIFIER[vocabulary]
	num_classes = len(metadata.thing_classes)
	reset_cls_test(predictor.model, classifier, num_classes)
	return predictor, metadata


def segmentation(image_path, predictor, metadata):
  im = cv2.imread(image_path)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  outputs = predictor(im)

  ## To print all masks 
  # #cv2_imshow(out.get_image()[:, :, ::-1])
  # #v = Visualizer(im[:, :, ::-1], metadata)
  # #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  
  # Code to get the classes detected - this returns the integer value
  classes = outputs["instances"].to("cpu").pred_classes.tolist()

  # Code to get the class label detected - this returns the string value given the integer
  class_names = [metadata.thing_classes[cl] for cl in classes]


  return outputs, class_names

