

import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from repvit_sam import SamPredictor
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2

from Dataset import ImageMaskDataset
from utility import dice_coefficient, sensitivity, specificity
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from modeling.build_sam import sam_model_registry

image_dirs_val = ["MICCAI/instrument_1_4_testing/instrument_dataset_4/left_frames"]
mask_dirs_val = ["MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_4/gt/BinarySegmentation"]
validation_transform = A.Compose([
    A.Resize(1024,1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

mask_transform  = transforms.Compose([

    transforms.Resize((1024,1024)),
    transforms.ToTensor()
])
datasetTest = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=validation_transform)
dataloaderTest = DataLoader(datasetTest,batch_size=2,shuffle=True)
def display_image(dataset, image_index):
    '''Display the image and corresponding three masks.'''

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for ax in axs.flat:
        ax.axis('off')

    # Display each image in its respective subplot
    axs[0, 0].imshow(dataset['train'][image_index]['image'])
    axs[0, 1].imshow(dataset['train'][image_index]['color_mask'])
    axs[1, 0].imshow(dataset['train'][image_index]['watershed_mask'])
    axs[1, 1].imshow(dataset['train'][image_index]['annotation_mask'])

    # Adjust spacing between images
    plt.subplots_adjust(wspace=0.01, hspace=-0.6)

    plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = np.array([coords[i] for i in range(len(coords)) if labels[i] == 1])
    # neg_points = np.array([coords[i] for i in range(len(coords)) if labels[i] == 0])
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    # ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_bbox(bbox, ax):
    for box in bbox:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def predict_boxes(predictor, boxes):


    masks, _, low_res = predictor.predict_torch(
        # predict_torch serve quando ho le bboxes altrimenti predictor.predict
        point_coords=None,
        point_labels=None,
        boxes=boxes,
        multimask_output=False,
    )
    return masks, _, low_res

def predict_points_boxes(predictor,boxes,centroids,input_label):
    all_masks = []
    all_scores = []
    all_low_res = []
    print(transformed_boxes.shape)
    print(input_label.shape)
    print(centroids.shape)

    # centroids: torch.Size([1, N, 2])
    # input_label: torch.Size([1, N])
    # boxes: torch.Size([num_boxes, 4]) — num_boxes > 1

    #devo creare le maschere dando in nput una box e un punto per volta perche non è possible dare piu boxe
    #e allo stesso tempo piu punti

    for i in range(boxes.shape[0]):
        box = boxes[i].unsqueeze(0)  # shape: [1, 4], batch size 1
        #print(box.shape)
        centroid = centroids[:,i,:].unsqueeze(0) # shape: [1,1,2]
        #print(centroid.shape)
        input = input_label[:,i].unsqueeze(0) # shape: [1, N]
        #print(input.shape)
        masks, scores, low_res = predictor.predict_torch(
            point_coords=centroid,
            point_labels=input,
            boxes=box,
            multimask_output=False
        )

        all_masks.append(masks)  # masks: [1, C, H, W]
        all_scores.append(scores)  # scores: [1, C]
        all_low_res.append(low_res)  # low_res: [1, C, H, W]

    # Concatenazione lungo la dimensione delle maschere (C)
    final_masks = torch.cat(all_masks, dim=0)  # [1, total_C, H, W]
    final_scores = torch.cat(all_scores, dim=0)  # [1, total_C]
    final_low_res = torch.cat(all_low_res, dim=0)

    return final_masks, final_scores, final_low_res
def calculate_iou(mask_pred, mask_gt):
    # Ensure the inputs are NumPy arrays
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.cpu().numpy()
    if isinstance(mask_gt, torch.Tensor):
        mask_gt = mask_gt.cpu().numpy()

    # Calculate the intersection (common pixels in both masks)
    intersection = np.logical_and(mask_pred, mask_gt).sum()

    # Calculate the union (all pixels that are 1 in at least one of the masks)
    union = np.logical_or(mask_pred, mask_gt).sum()

    # Calculate IoU (Intersection over Union)
    iou = intersection / union if union != 0 else 0  # Avoid division by zero

    return iou

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def refining(mask):
    # 1. Rimuovi rumore (morphological opening)
    #mask = mask.detach().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    while mask.ndim > 2:
        mask = mask[0]
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 2. Chiudi buchi interni (closing)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 3. (opzionale) Gaussian blur per bordi morbidi
    mask_blurred = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    mask_blurred = mask_blurred/255

    return mask_blurred




device = "cuda" if torch.cuda.is_available() else "cpu"

student_checkpoint = "checkpoints/13_05/decoupledVitBDGfFE.pth"

model = sam_model_registry["CMT"](checkpoint=student_checkpoint)
model.to(device=device)
#model.load_state_dict(state_dict)
#print("Missing keys:", model.load_state_dict(state_dict, strict=False))
#CARICO UN MODELLO SAM
#sam_checkpoint = "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/checkpoints/sam_vit_b_01ec64.pth"
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"



sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
#ASSEGNO L'IMAGE ENCODER DISTILLATO A SAM
#sam.image_encoder = model.image_encoder
sam.eval()
model.eval()
predictor = SamPredictor(sam)
#print("State dict keys:", state_dict.keys())
"""
checkpoint = torch.load("C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/checkpoints/student_checkpoint.pth", map_location="cpu")
 
image_encoder_state_dict = {
    k.replace("image_encoder.", ""): v
    for k, v in checkpoint.items()
    if k.startswith("image_encoder.")
}

model.image_encoder.load_state_dict(image_encoder_state_dict)
sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")
transformer_dim = model.mask_decoder.transformer_dim
transformer = model.mask_decoder.transformer


cloned_mask_decoder = type(sam.mask_decoder)(transformer_dim=transformer_dim, transformer=transformer)
cloned_mask_decoder.load_state_dict(sam.mask_decoder.state_dict())  # Copy the weights
model.mask_decoder = cloned_mask_decoder
cloned_prompt_encoder = copy.deepcopy(sam.prompt_encoder)
 # Copy the weights

# Assign the cloned prompt encoder to the model
model.prompt_encoder = cloned_prompt_encoder 
"""
#model.to(device=device)
#model.eval()



#predictor = SamPredictor(model)
timeDf = pd.DataFrame(columns=['time', 'index', 'iou','dice','sensitivity','specificity'])

for images, labels in dataloaderTest:  # i->batch index, images->batch of images, labels->batch of labels



        images = images.to(device)
        print(images.shape)
        labels = labels.to(device)
        print(labels.shape)
        results_teach = []
        results_stud = []

        for image, label in zip(images, labels):
            # Sposta su CPU e converti in NumPy

            label = label.detach().cpu().numpy()

            image = image.detach().cpu().numpy()

            # Rimuovi batch e channel dimension se presenti (es: (1, 1024, 1024))
            label = np.squeeze(label)
            image = np.transpose(np.squeeze(image), (1, 2, 0))  # (C,H,W) → (H,W,C)
            #print(image.shape)
            # Binarizza la maschera
            label_bin = (label > 0).astype(np.uint8)
            # Convert to binary mask
            contours, _ = cv2.findContours(label_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            # print("contours",contours)

            centroids = []
            input_label = []
            bbox = []
            if contours:
                for countour in contours:
                    M = cv2.moments(countour)
                    if M["m00"] != 0:
                        centroid_x = int(M["m10"] / M["m00"])
                        centroid_y = int(M["m01"] / M["m00"])
                        centroids.append([centroid_x, centroid_y])
                        input_label.append(1)
                        x, y, w, h = cv2.boundingRect(countour)
                        bbox.append([x, y, x + w, y + h])
            centroids = torch.tensor(centroids,device=device)
            print(centroids)

            bbox = torch.tensor(bbox).float()
            original_size = tuple(map(int, images[0].shape[-2:]))
            transformed_boxes = predictor.transform.apply_boxes_torch(bbox, (1024,1024))
            centroids = predictor.transform.apply_coords_torch(centroids, (1024, 1024)).unsqueeze(0)
            input_label = torch.tensor(input_label, dtype=torch.int64).unsqueeze(0)
            #image = np.transpose(image,(1,2,0))
            image = (image * 0.5 + 0.5) * 255
            image = image.astype(np.uint8)
            print(image.shape)
            print("Image shape:", image.shape)
            print("Image min/max values:", image.min(), image.max())
            #plt.imshow(image.permute(1, 2, 0))
            start_time = time.time()

            predictor.set_image(image)
            masks, _, low_res = predict_points_boxes(predictor, transformed_boxes, centroids, input_label)
            end_time = time.time()
            maskunion = np.zeros_like(masks[0].cpu().numpy())
            for mask in masks:
                mask = mask.detach().cpu().numpy()
                mask = refining(mask)


                #show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                #values, counts = np.unique(mask.cpu().numpy(), return_counts=True)
                #print("unique", values)
                #print("counts", counts)


                maskunion = np.logical_or(maskunion, mask)


            latency = (end_time - start_time) * 1000
            iou = calculate_iou(maskunion, label)
            dice = dice_coefficient(maskunion, label)
            sens = sensitivity(maskunion, label)
            spec = specificity(maskunion, label)

            timeDf.loc[len(timeDf)] = [latency, len(timeDf), iou,dice,sens,spec]
timeDf.to_csv('RISULTATI/TimeDfBBoxStudent.csv', index=False)
print(timeDf)

