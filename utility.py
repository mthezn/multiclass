import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
import random
import string
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from skimage import measure
def cutmix_with_mask(imgs, masks, labels, alpha=1.0):
    # imgs: (B,3,H,W)
    # masks: (B,1,H,W)
    # labels: (B,) interi

    B, _, H, W = imgs.shape
    lam = np.random.beta(alpha, alpha)

    # scegli due indici random
    idx = torch.randperm(B)
    img2, mask2, label2 = imgs[idx], masks[idx], labels[idx]

    # genera bounding box di cutmix
    cx, cy = np.random.randint(W), np.random.randint(H)
    w = int(W * np.sqrt(1 - lam))
    h = int(H * np.sqrt(1 - lam))
    x1, y1 = np.clip(cx - w // 2, 0, W), np.clip(cy - h // 2, 0, H)
    x2, y2 = np.clip(cx + w // 2, 0, W), np.clip(cy + h // 2, 0, H)

    # sostituisci sia immagine sia mask
    imgs[:, :, y1:y2, x1:x2] = img2[:, :, y1:y2, x1:x2]
    masks[:, :, y1:y2, x1:x2] = mask2[:, :, y1:y2, x1:x2]

    # mescola etichette (soft labels)
    labels_onehot = F.one_hot(labels, num_classes=9).float()
    labels2_onehot = F.one_hot(label2, num_classes=9).float()
    new_labels = lam * labels_onehot + (1 - lam) * labels2_onehot

    return imgs, masks, new_labels

def save_binary_mask(mask_tensor, epoch, batch_idx, output_dir="binary_masks"):
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy array
    if isinstance(mask_tensor, torch.Tensor):
        mask_np = mask_tensor.detach().cpu().numpy()
    else:
        mask_np = mask_tensor

    # Ensure shape
    if mask_np.ndim == 4:
        mask_np = mask_np[0, 0]
    elif mask_np.ndim == 3:
        mask_np = mask_np[0]

    # Ensure binary uint8 image
    mask_np = (mask_np > 0).astype(np.uint8) * 255

    # Save
    Image.fromarray(mask_np).save(
        os.path.join(output_dir, f"mask_epoch{epoch}_batch{batch_idx}.png")
    )

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
def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def predict_boxes(predictor,image, boxes):
    image_array = (image[0].detach().cpu().numpy())
    image_array = np.transpose(image_array, (1, 2, 0))
    image = (image_array * 0.5 + 0.5) * 255
    image = image.astype(np.uint8)  # Normalizza l'immagine per visualizzazione
    predictor.set_image(image)

    masks, _, low_res = predictor.predict_torch(
        # predict_torch serve quando ho le bboxes altrimenti predictor.predict
        point_coords=None,
        point_labels=None,
        boxes=boxes,
        multimask_output=False,
    )
    return masks, _, low_res


def predict_points_boxes_manual(model, image_embedding, boxes, centroids, input_label):
    all_masks = []
    all_scores = []
    all_low_res = []
    model_device = next(model.prompt_encoder.parameters()).device


    for i in range(boxes.shape[0]):
        # Estrai singola box, punto e label
        box = boxes[i].unsqueeze(0).to(device=model_device) # [1, 4]
        point = centroids[:, i, :].unsqueeze(0).to(device=model_device) # [1, 1, 2]
        label = input_label[:, i].unsqueeze(0).to(device = model_device) # [1, 1]

        # Encode prompt: box + point
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(point, label),
            boxes=box,
            masks=None,
        )

        # Usa mask decoder
        low_res_logits, score = model.mask_decoder(
            image_embeddings=image_embedding,          # [1, C, H', W']
            image_pe=model.prompt_encoder.get_dense_pe(),  # positional encoding
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # Upscale maschera a risoluzione originale
        mask = model.postprocess_masks(low_res_logits, input_size=(image_embedding.shape[-2], image_embedding.shape[-1]),original_size=(1024, 1024))

        all_masks.append(mask)
        all_scores.append(score)
        all_low_res.append(low_res_logits)

    # Concatenazione dei risultati
    if all_masks == []:
        return torch.zeros((1, 1, 1024, 1024)).to(device=model_device), torch.zeros((1, 1)).to(device=model_device), torch.zeros((1, 1, 1024, 1024)).to(device=model_device)
    final_masks = torch.cat(all_masks, dim=0)  # [N, 1, H, W]
    final_scores = torch.cat(all_scores, dim=0)  # [N, 1]
    final_low_res = torch.cat(all_low_res, dim=0)

    return final_masks, final_scores, final_low_res
def predict_points_boxes(predictor,image,boxes,centroids,input_label):
    all_masks = []
    all_scores = []
    all_low_res = []
    image_array =(image[0].detach().cpu().numpy())
    image_array = np.transpose(image_array, (1, 2, 0))
    image = (image_array*0.5 + 0.5 )*255
    image = image.astype(np.uint8)# Normalizza l'immagine per visualizzazione
    predictor.set_image(image)
    model_device = next(predictor.model.parameters()).device  # Assicura coerenza col modello

    for i in range(boxes.shape[0]):
        box = boxes[i].unsqueeze(0).to(model_device)  # shape: [1, 4]
        centroid = centroids[:, i, :].unsqueeze(0).to(model_device)  # shape: [1, 1, 2]
        label = input_label[:, i].unsqueeze(0).to(model_device)  # shape: [1, 1]

        masks, scores, low_res = predictor.predict_torch(
            point_coords=centroid,
            point_labels=label,
            boxes=box,
            multimask_output=False
        )

        all_masks.append(masks)
        all_scores.append(scores)
        all_low_res.append(low_res)
    if all_masks == []:
            return torch.zeros((1, 1, 1024, 1024)).to(device=model_device), torch.zeros((1, 1)).to(
                device=model_device), torch.zeros((1, 1, 1024, 1024)).to(device=model_device)
    # Concatenazione dei risultati
    final_masks = torch.cat(all_masks, dim=0)
    final_scores = torch.cat(all_scores, dim=0)
    final_low_res = torch.cat(all_low_res, dim=0)

    return final_masks, final_scores, final_low_res

def get_bbox_centroids(label,num_box=2):
    """
    Function: get_bbox_centroids

    Purpose:
        Extracts the bounding boxes and centroids of the largest contours (connected components)
        found in a binary mask label. This is useful for identifying object locations and
        preparing inputs for models that require point and box prompts.

    Arguments:
        label (np.ndarray):
            A 2D numpy array representing a binary mask (or multi-class mask) where objects
            are segmented. Shape is typically (H, W).

        num_box (int, optional, default=2):
            The maximum number of largest contours to extract. Contours are sorted by area
            in descending order, and only the top `num_box` are returned.

    Returns:
        centroids (np.ndarray):
            An array of shape (num_box, 2) containing the (x, y) coordinates of the centroids
            for each extracted contour.

        bbox (list of lists):
            A list of bounding boxes corresponding to each contour, each represented as
            [x_min, y_min, x_max, y_max].

        input_label (list of int):
            A list of labels (all ones) corresponding to the extracted contours, typically
            used as positive prompts for segmentation models.


    """
    # Create contours from the gt
    contours, _ = cv2.findContours(label.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_box]

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
    centroids = np.array(centroids)



    return centroids,bbox,input_label


def generate_random_name(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def contains_instrument(example):
    mask = np.array(example["color_mask"])  # o "segmentation" se diverso
    return np.any((mask == 169) | (mask == 170))

def refining(mask):
    # 1. Rimuovi rumore (morphological opening)
    #mask = mask.detach().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    while mask.ndim > 2:
        mask = mask[0]

    #cleaned_mask = np.zeros_like(mask)
    #num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Mantieni solo i componenti connessi con area >= min_area
    #for i in range(1, num_labels):  # Salta lo sfondo (etichetta 0)
     #   if stats[i, cv2.CC_STAT_AREA] >= 100:
      #      cleaned_mask[labels == i] = 255

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 2. Chiudi buchi interni (closing)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 3. (opzionale) Gaussian blur per bordi morbidi
    mask_blurred = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    mask_blurred = mask_blurred/255

    return mask_blurred


def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the Dice Coefficient between two binary masks.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()

    if total == 0:
        return 1.0  # both empty, perfect match
    return 2.0 * intersection / total


def sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Sensitivity (Recall) = TP / (TP + FN)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()

    if tp + fn == 0:
        return 1.0  # no positive cases in ground truth
    return tp / (tp + fn)


def specificity(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Specificity = TN / (TN + FP)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    tn = np.logical_and(np.logical_not(pred), np.logical_not(target)).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()

    if tn + fp == 0:
        return 1.0  # no negative cases in ground truth
    return tn / (tn + fp )

def masked_global_avg_pool(fseg: torch.Tensor, masks: torch.Tensor):
    """
    fseg: [C, H, W] feature map dal decoder binario
    masks: [N, H, W] maschere binarie delle istanze (0/1)

    Ritorna:
        pooled: [N, C] vettori mediati sulle aree delle istanze
    """
    N, H, W = masks.shape
    C = fseg.shape[0]

    masked_feats = fseg.unsqueeze(0) * masks.unsqueeze(1)   # [N, C, H, W]
    masked_feats = masked_feats.view(N, C, -1)              # [N, C, H*W]

    sums = masked_feats.sum(-1)                             # [N, C]
    areas = masks.view(N, -1).sum(-1).clamp(min=1).unsqueeze(1)  # [N,1]

    pooled = sums / areas
    return pooled

def extract_instances(mask_pred, min_area=100):
    """
    mask_pred: Tensor [H, W] binaria
    ritorna lista di maschere per ogni istanza (connected components).
    """
    mask_np = mask_pred.cpu().numpy().astype("uint8")
    labeled = measure.label(mask_np)  # connected components
    instances = []
    for region in measure.regionprops(labeled):
        if region.area >= min_area:
            inst_mask = (labeled == region.label).astype("uint8")
            instances.append(torch.from_numpy(inst_mask))
    return instances



def get_instance_labels(gt_mask: np.ndarray, inst_masks: np.ndarray):
    """
    Args:
        gt_mask: [H,W] con classi (0=background, 1..N strumenti)
        inst_masks: [N,H,W] array di maschere binarie (0/1) per ogni istanza

    Returns:
        labels: lista di lunghezza N con l'ID di classe per ogni istanza
    """
    labels = []
    for inst_mask in inst_masks:
        # prendi i pixel GT sotto la maschera
        #print(gt_mask.shape)
        gt_vals = gt_mask[inst_mask > 0]

        # ignora background
        gt_vals = gt_vals[gt_vals > 0]

        if len(gt_vals) == 0:
            # nessuna sovrapposizione → istanza vuota o rumore
            labels.append(0)  # oppure 0 se vuoi dire "background"
        else:
            # classe dominante
            class_id = np.bincount(gt_vals).argmax()
            labels.append(class_id)

    return labels