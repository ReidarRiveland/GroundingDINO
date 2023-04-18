import os
import torch
import torchvision.transforms.functional as F
from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv

HOME = os.getcwd()

def dino_process_img(image): 
    CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

    WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
    print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

    TEXT_PROMPT = 'animals'
    BOX_TRESHOLD = 0.4
    TEXT_TRESHOLD = 0.4

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )
    return boxes, logits, phrases


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

IMAGE_PATH = 'stimuli/Training_Stim/Animals/Pair4/219.jpg'
image_source, image = load_image(IMAGE_PATH)

import matplotlib.pyplot as plt
plt.imshow(  image.permute(1, 2, 0)  )
plt.show()

boxes, logits, phrases = dino_process_img(image)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
sv.plot_image(annotated_frame, (16, 16))

pixeled = extract_bbox_pixels(image_source, boxes)

from torchvision.utils import draw_segmentation_masks
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torchvision.io import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
model = model.eval()



output = model(images)

