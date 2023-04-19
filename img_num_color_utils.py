import os
import numpy as np
from numba import *

import torch
from torchvision.utils import draw_segmentation_masks
from torchvision.io import read_image
import torchvision.transforms.functional as F
import torchvision.transforms as T

from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.getcwd()

def dino_process_img(image, text_prompt, box_threshold = 0.4, text_threshold = 0.4): 
    CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

    WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
    print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    model.eval()
    model.to(device)

    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=text_prompt, 
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    return boxes, logits, phrases

def get_semantic_segment(image_list, score_threshold=0.4, prob_threshold=0.5):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()

    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    model.eval()
    model.to(device)

    batched = [transforms(Image.fromarray(img)).to(device) for img in image_list]
    output = model(batched)

    boolean_masks = [
        (out['masks'].cpu()[out['scores'].cpu()>score_threshold] > prob_threshold).squeeze(1).sum(0).bool()
        for out in output
        ]

    imgs_with_masks = [
        draw_segmentation_masks(torch.tensor(img.transpose(2, 0, 1), dtype=torch.uint8), mask, alpha=0.8)
        for img, mask in zip(image_list, boolean_masks)
        ]

    segmented_pixels = [img[mask, :] 
        for img, mask in zip(image_list, boolean_masks)
        if not torch.all(mask==False)
        ]

    return segmented_pixels, imgs_with_masks, output

def extract_bbox_pixels(img, bboxes): 
    extracted_pixels = []
    for i in range(bboxes.shape[0]):
        box = bboxes[i, :]
        H = img.shape[0]
        W = img.shape[1]

        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        extracted_pixels.append(img[y0:y1,x0:x1,:])

    return extracted_pixels


@njit(parallel=True)
def _image_colorfulness(rgb_pixels):
    R = rgb_pixels[..., 0]
    G = rgb_pixels[..., 1]
    B = rgb_pixels[..., 2]
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute((0.5 * (R + G)) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    rgMean, rgStd = np.mean(rg), np.std(rg)
    ybMean, ybStd = np.mean(yb), np.std(yb)
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return [stdRoot + (0.3*meanRoot), rgMean, rgStd, ybMean, ybStd]

def image_colorfulness(all_segmented_pixels): 
    return np.mean(np.array([_image_colorfulness(p) for p in all_segmented_pixels]), axis=0)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def detect_objects(image_source, image, text_prompt, show_detections, **dino_kwargs):
    boxes, logits, phrases = dino_process_img(image, text_prompt=text_prompt, **dino_kwargs)
    if show_detections: 
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        sv.plot_image(annotated_frame, (16, 16))
    choose_box_list = [phrase != '' for phrase in phrases]
    pixeled = extract_bbox_pixels(image_source, boxes[choose_box_list, :])

    return len(pixeled), pixeled

def detect_colorfulness(pixels_list, show_segments, **segment_kwargs):
    segmented_pixels, imgs_with_masks, _ = get_semantic_segment(pixels_list, **segment_kwargs)
    if show_segments: 
        show(imgs_with_masks)
        plt.show()
    return image_colorfulness(segmented_pixels)

def get_animal_num_color(image_path, show_detections=True, show_segments=True, **dino_kwargs): 
    image_source, image = load_image(image_path)

    num_objects, pixels  = detect_objects(image_source, image, 'animals', show_detections=show_detections, box_threshold=0.4, text_threshold=0.4)
    color_score = detect_colorfulness(pixels, show_segments=show_segments)

    return num_objects, color_score

def get_car_color(image_path, show_segments=True, **segment_kwargs):
    image_source, image = load_image(image_path)
    color_score = detect_colorfulness([image_source], show_segments=show_segments, **segment_kwargs)
    return color_score

def get_fruit_num(image_path, show_detections=True, **dino_kwargs): 
    image_source, image = load_image(image_path)

    num_objects, _  = detect_objects(image_source, image, 'individual vegetable.', show_detections=show_detections, **dino_kwargs)

    return num_objects



# from PIL import ImageEnhance
IMAGE_PATH = 'stimuli/Training_Stim/Cars/Pair1/313.jpg'
image_source, image = load_image(IMAGE_PATH)

plt.imshow(image_source)
plt.show()

# import colorsys

# colorsys.rgb_to_hls(r, g, b)


# test_enhanced_img_color(IMAGE_PATH, ImageEnhance.Contrast, 5.0)





###IS THE COLOR SCALE THE SAME



import warnings
from transformers import logging
logging.set_verbosity_error()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # IMAGE_PATH = 'stimuli/Training_Stim/Animals/Pair1/213.jpg'
    # _, col = get_animal_num_color(IMAGE_PATH, show_segments=False)
    # col

    # IMAGE_PATH = 'stimuli/Training_Stim/Animals/Pair1/214.jpg'
    # _, col = get_animal_num_color(IMAGE_PATH, show_segments=False)
    # col

    IMAGE_PATH = 'stimuli/Training_Stim/Cars/Pair1/313.jpg'
    col = get_car_color(IMAGE_PATH)
    col

    IMAGE_PATH = 'stimuli/Training_Stim/Cars/Pair1/313.jpg'
    col = get_car_color(IMAGE_PATH)
    col