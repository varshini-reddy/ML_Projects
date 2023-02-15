
import PIL 
import cv2
import numpy as np
import json 
import matplotlib.pyplot as plt


def image_blend(background, img, mask, x, y):
    '''
    Arguments:
    background - background image in CV2 RGB format
    img - image of object in CV2 RGB format
    mask - mask of object in CV2 RGB format
    x, y - coordinates of the center of the object image
    0 < x < width of background
    0 < y < height of background
    
    Function returns background with added object in CV2 RGB format
    
    CV2 RGB format is a numpy array with dimensions width x height x 3
    '''
    
    bg = background.copy()
    
    h_bg, w_bg = bg.shape[0], bg.shape[1]
    
    h, w = img.shape[0], img.shape[1]
    
    # Calculating coordinates of the top left corner of the object image
    x = x - int(w/2)
    y = y - int(h/2)    
    
    mask_boolean = mask[:,:,0] == 0
    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)
    
    if x >= 0 and y >= 0:
    
        h_part = h - max(0, y+h-h_bg) # h_part - part of the image which overlaps background along y-axis
        w_part = w - max(0, x+w-w_bg) # w_part - part of the image which overlaps background along x-axis

        bg[y:y+h_part, x:x+w_part, :] = bg[y:y+h_part, x:x+w_part, :] * ~mask_rgb_boolean[0:h_part, 0:w_part, :] + (img * mask_rgb_boolean)[0:h_part, 0:w_part, :]
        
    elif x < 0 and y < 0:
        
        h_part = h + y
        w_part = w + x
        
        bg[0:0+h_part, 0:0+w_part, :] = bg[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[h-h_part:h, w-w_part:w, :] + (img * mask_rgb_boolean)[h-h_part:h, w-w_part:w, :]
        
    elif x < 0 and y >= 0:
        
        h_part = h - max(0, y+h-h_bg)
        w_part = w + x
        
        bg[y:y+h_part, 0:0+w_part, :] = bg[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[0:h_part, w-w_part:w, :] + (img * mask_rgb_boolean)[0:h_part, w-w_part:w, :]
        
    elif x >= 0 and y < 0:
        
        h_part = h + y
        w_part = w - max(0, x+w-w_bg)
        
        bg[0:0+h_part, x:x+w_part, :] = bg[0:0+h_part, x:x+w_part, :] * ~mask_rgb_boolean[h-h_part:h, 0:w_part, :] + (img * mask_rgb_boolean)[h-h_part:h, 0:w_part, :]
    
    return bg

