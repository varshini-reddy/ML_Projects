The notebook `Inference_Inpaint.ipynb` has the entire pipeline to save results from inpainting and inpainting with displacement. 

The following are the sections and functions within the notebook:

## Libraries and Requirements 

- This section has the code to download the zip files and install all required libraries.
- Model weights of the memorability model and the LaMa model is stored in the zipfiles as well. 
- If you would like to run it on any other platform, you would have to manually download the zipfiles and then pip install the required modules. For most systems, this would have to be done only once.



## Object Segmentation

### `get_predictor( )`

- This function takes no input.
- It is used to load the Detic model. The model weights are loaded from the zip files.
- To change the vocabulary, a numpy file needs to be defined in `BUILDIN_CLASSIFIER` and update the variable `vocabulary` 



### `segmentation( )`

- This function takes as input:
  - `img` - This can be image or image file path
  -  `input_type` - The toggle between image and file path is handled by this. `img` by default to get image. Any other keyword would result the function considering the input to be only image path.
- It returns the masks and mask names that have been detected based on the vocabulary.



## Memorability Model

- This section loads the memorability model.
- The variable `model` is the loaded model



## Image Transformations

### `ImageTransform( )`

- This function takes as input:
  - `img` - This can be image, numpy array or image path.
  - `in_type` - The toggle between different image format inputs is handledby this. `filename` by default expecting an image path. The keyword `otr` assumes that the input is an image read in as a numpy array. Any other keyword would assume that the input image is in `PIL` format.

- The output of this function is a transformed image that can be used to query the memorability model. All images are transformed to a PyTorch tensor of size 456x456



### `alter_mask( )`

- This function takes as input:
  - `mask` - A mask of any size. 
  - `val1` - Optional parameter, default to 20. It is used to control the size to open a mask
  - `val2` - Optional parameter, default to 50. It is used to control the size of mask dilation. 

- This returns an open+dilated image in OpenCV format.



## Image Blending

### `image_blend( )`

- This function takes as input:
  - `background` - background image in CV2 RGB format
  - `img` - image of object in CV2 RGB format
  - `mask` - mask of object in CV2 RGB format
  - `x`, `y` - coordinates of the center of the object image

- It outputs one image with the object position in the specified location



## In-painting Model

### `inpaint( )`

- This function takes as input:
  - `model` - This is the LaMa inpaint model that has been loaded from the zipfiles, saved as `inpaint_model`
  - `img` - The Pytorch tensor input image to be in-painted
  - `mask` - A numpy array of masks
- The output is an inpainted image



### `postprocess( )`

- This function takes as input an image as a PyTorch tensor 
- The output is a processed image to compliment the in-painting model



## Mask Validity

### `check_mask_validity( )`

- This function takes as input: 
  - `mask` : A mask as a numpy array of 0s and 255s
- The output is a boolean value which states `True` if the mask is valid i.e. more than 5% and less than 80% of the image and `False` otherwise.



## Clip Embedding

### `get_clip_embedding( )`

- This function takes as input:
  - `img` - Takes image or image path as input. 
  - `input_type` - This is used to toggle between image and image path. `otr` which is default expects an image. `org` can be used to pass a image file path.
- The output is a `(512,)` numpy array of the clip embeddings for the input image

- Additionally, the `clip_model` defined loads the clip embedding model



## Inference Text

### `inference_save( )`

- This function takes as input:
  - `original_prediction` - A list consisting of all predictions for the original prediction. The first 5 values are the memorability scores and the next 14 are the saliency heat maps. These values remain constant for an image regardless of the mask in question.
  - `inpaint_prediction` - A lists consisting of all predictions for each image inpainted for a particular mask. The first 5 values are the memorability scores and the next 14 are the saliency heat maps.
  - `inpaint_displace_prediction` - A dictionary of lists. The keys of the dictionary can be used to access the results for each of the position the mask has been displaced to. In this notebook, this has the folllowing keys `Top_Left, Top_Right, Bottom_Left, Bottom_Right, Center`. Each of the value in this dictionary has 5 memorability scores and 14 saliency heat maps.
  - `original_clip_embedding` - Clip embedding for the original image
  - `clip_embedding` - A list of clip embeddings. The first the clip embedding for the in-painted image of the current mask followed by 5 clip embeddings for inpaint+displacement image versions for the current mask.
  - `image_name` - This is the name of the original input image file.
  - `class_name` - This is the name of the mask that is currently being saved
  - `dpos` - This is a dictionary where the keys are the position names and values are tuples of format `(width, height)` for image displacement
- This function also assumes some global variables defined:
  - `compiled_result_list` - List to store dictionary of results for the current mask. This is used to save the final csv. 
  - `inpaint_count` - Integer variable intially defined as 0 to save heatmap outputs for the in-painted versions of the image
  - `inpaint_disp_count` - Integer variable intially defined as 0 to save heatmap outputs for the in-painted + displaced versions of the image

- This function has no output. It automatically saves the heat map as images and updated the global csv variable.



## Main Function 

### `main( )`

- This function takes as input:

  - `img_name` - Ths is the image filename
  - `img_path` - This is the path to the folder holding the image file

- It calls all the functions defined above in the following sequence:

  - `segmentation` to get all masks of the original image
  - `get_clip_embedding` to get clip embedding of the original image
  - `ImageTransform` to transform the input image to a format required by the memorability model
  - `alter_mask` to alter the mask to open and dilate it
  - `inpaint` to occlude the current mask by inpainting
  - `get_clip_embedding` to get clip embedding of the inpainted image
  - `image_blend` to add the mask object back to the image at the desired position
  - `get_clip_embedding` to get clip embedding of the inpainted + displaced image
  - `inference_save` to update the csv results and save the heatmaps  

  





















