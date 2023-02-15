import pandas as pd
from PIL import Image
import clip
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip
from torchvision import transforms
import torch
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_encoded_frame(image, model):
    transform_image = transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Resize((224,224))
                                        ])
    image = transform_image(image)
    #plt.imshow(image.permute(1, 2, 0))
    image = image.unsqueeze(dim=0)

    with torch.no_grad():
        encoded_image = model.encode_image(image.cuda()).squeeze()
    
    return encoded_image


def get_encoded_image( image_folder, model):
    image = Image.open(image_folder).convert('RGB')
    transform_image = transforms.Compose([
                                            transforms.PILToTensor(),
                                            transforms.Resize((224,224))
                                         ])
    image = transform_image(image)
    #plt.imshow(image.permute(1, 2, 0))
    image = image.unsqueeze(dim=0)

    with torch.no_grad():
        encoded_image = model.encode_image(image.cuda()).squeeze()
    
    return encoded_image


def find_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    return scene_manager.get_scene_list()



def generate_clip_embeddings(file_path, file_type, model):

    if file_type == "mp4":
        print("True for mp4")

        scenes = find_scenes(file_path)
        timestamps = []
        for scene in scenes:
            timestamps.append((scene[0].get_timecode(), scene[1].get_timecode()))

        vfclip = VideoFileClip(file_path)
        encoded_frames = []
        if len(timestamps) == 1:
            # Videos with only one scene
            tsamp0, tsamp1 = timestamps[0]
            video = vfclip.subclip(tsamp0, tsamp1)
            duration = video.duration//2
            frame = video.get_frame(duration)
            encoded_frame = get_encoded_frame(frame, model)
            encoded_frames.append(encoded_frame)

        else:
            # Videos with more than one scene
            frame_step = 25
            for i, frame in enumerate(vfclip.iter_frames()):
                if i % frame_step != 0:
                    continue
                encoded_frame = get_encoded_frame(frame,model)
                encoded_frames.append(encoded_frame)
        
        encoded_features_tensor = torch.stack(encoded_frames).mean(dim=0)
        encoded_features_df = pd.DataFrame(encoded_features_tensor.tolist())

    elif file_type == "jpg" or file_type=='png' or file_type=='jpeg':
        encoded_image = get_encoded_image(file_path, model)
        encoded_features_df = pd.DataFrame(encoded_image.tolist())

    else:
        return 'Unhandled option'
 
    encoded_data = encoded_features_df.transpose()
    # new_col_names = {}
    # for i in range(512):
    #     new_col_names[i] = f"CLIP_{i}"
    # encoded_data.rename(columns=new_col_names, inplace=True)

    # encoded_data['Ad ID'] = adid
        
    return encoded_data


def get_clip_embeddings(filename, org_path=""):
    
    # downloaded_ads, downloaded_formats = get_downloaded_ad_id_list(path_folder)
    # df = get_df_downloaded_ads(df, downloaded_ads)
    # ads_dictionary = make_ads_dictionary(downloaded_ads, downloaded_formats)
    path_folder = os.path.join(org_path, filename)
    file_type = filename.split(".")[1]
    encoded_df = generate_clip_embeddings(path_folder, file_type, model)
    
    return encoded_df





