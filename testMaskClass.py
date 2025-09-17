import torch
import numpy as np
import cv2
import os
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
from datasets import load_dataset

from Dataset import ImageMaskDataset, CholecDataset, Kvasir, InstrumentDataset
import torch
from torch.utils.data import DataLoader

from instrumentClassifier import InstrumentClassifier
from utility import extract_instances, masked_global_avg_pool, get_instance_labels
from modeling.build_sam import sam_model_registry
from utility import dice_coefficient,sensitivity,specificity
import os

CLASS_COLORS = {
    0: (0, 0, 0),          # background = nero
    1: (255, 0, 0),        # Large_Needle_Driver= rosso
    2: (0, 255, 0),        # Prograsp_Forceps= verde
    3: (0, 0, 255),        # Bipolar_Forceps = blu
    4: (255, 255, 0),      # Grasping_Retracto = giallo
    5: (255, 0, 255),      # Maryland_Bipolar_Forceps = magenta
    6: (0, 255, 255),      # Monopolar_Curved_Scissors = ciano
    7: (128, 0, 128),      # Other = viola
    8: (255, 165, 0),      # Vessel_Sealer = arancione

}

def mask_to_rgb(mask, class_colors=CLASS_COLORS):
    """Converte una maschera di classi in immagine RGB con colori fissi"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in class_colors.items():
        rgb[mask == cls] = color
    return rgb

class_names = {
    1: 'Large_Needle_Driver',
    2:'Prograsp_Forceps',

    3: 'Bipolar_Forceps' ,
    4: 'Grasping_Retractor',
    5: 'Maryland_Bipolar_Forceps' ,
     6: 'Monopolar_Curved_Scissors',
     7: 'Other',
     8 : 'Vessel_Sealer'
}

GLOBAL_CLASS_MAPPING = {
    'Large_Needle_Driver': 1,
    'Prograsp_Forceps': 2,

    'Bipolar_Forceps': 3,
    'Grasping_Retractor': 4,
    'Maryland_Bipolar_Forceps': 5,
    'Monopolar_Curved_Scissors': 6,
    'Other': 7,
    'Vessel_Sealer': 8
}

image_dirs_val = ["MICCAI/instrument_1_4_testing/instrument_dataset_4/left_frames"]
mask_dirs_val = ["MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_2/TypeSegmentationRescaled"]

image_dirs_train = [

    "MICCAI/instrument_1_4_training/instrument_dataset_4/left_frames",
]
mask_dirs_train = [
    "MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth"]

validation_transform = A.Compose([
    A.Resize(1024,1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

#print(len(datasetKvasir))


def contains_instrument(example):
    mask = np.array(example["color_mask"])  # o "segmentation" se diverso
    return np.any((mask == 169) | (mask == 170))

datasetTest = InstrumentDataset(image_dirs=image_dirs_train, gt_dirs=mask_dirs_train, transform=validation_transform, class_to_id=GLOBAL_CLASS_MAPPING)
#datasetTest = CholecDataset(hf_dataset=filtered_ds, transform=validation_transform)
dataloaderTest = DataLoader(datasetTest, batch_size=2, shuffle=True)




# CARICO UN MODELLO SAM
# sam_checkpoint = "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/checkpoints/sam_vit_b_01ec64.pth"
autosam_checkpoint = "/home/mdezen/distillation/checkpoints/28_07/autoSamFineUnetMUcH0.pth"
model_type = "autoSamUnet"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = sam_model_registry[model_type](checkpoint=None, num_classes=1)
state_dict = torch.load(autosam_checkpoint, map_location=torch.device('cuda'))
model.load_state_dict(state_dict, strict=False)  # Load the state dict into the model
model.to(device=device)


model.eval()
classifier = InstrumentClassifier(in_channels=3, n_classes=9)
classifier_checkpoint = "checkpoints/01_09/classifieriXiAq.pth"
state_dict = torch.load(classifier_checkpoint)
classifier.load_state_dict(state_dict)
classifier.to(device=device)
classifier.eval()
out_dir="debug_samples"
max_batches=2


os.makedirs(out_dir, exist_ok=True)
model.eval()
classifier.eval()

colors = {}
rng = np.random.RandomState(42)

with torch.no_grad():
    for bi, (images, masks_gt) in enumerate(dataloaderTest):
        #if bi >= max_batches:
         #   break

        images = images.to(device)
        B, _, H, W = images.shape

        # 1. Segmentazione binaria
        feats = model.image_encoder(images)

        pred_logits,dict = model.mask_decoder(feats)
        pred_logits = model.postprocess_masks(pred_logits,(1024,1024), (H, W))

        pred_masks = pred_logits
        pred_masks = (pred_masks) > 0  # (B,1,H,W)
        inst_imgs, inst_masks, inst_labels = [], [], []
        for b in range(B):
            img_np = ((images[b].permute(1,2,0).cpu().numpy() *0.5 + 0.5)* 255).astype(np.uint8)
            img_np = np.ascontiguousarray(img_np)
            inst_masks_list = extract_instances(pred_masks[b,0])

            overlay = img_np.copy().astype(np.float32)
            for  inst_mask in inst_masks_list:
                mask_bool = inst_mask.cpu().numpy().astype(bool)

                mask_t = inst_mask.unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)
                img_t = images[b].unsqueeze(0)  # (1,3,H,W)
                logits = classifier(img_t, mask_t)
                pred_class = torch.argmax(logits, dim=1).item()

                color = np.array(CLASS_COLORS[pred_class], dtype=np.float32)

                overlay[mask_bool] = 0.5 *color +(1 - 0.5) * overlay[mask_bool]
                # centro per label
                ys, xs = np.where(mask_bool)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(
                        overlay,
                        class_names.get(pred_class, str(pred_class)),
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                # salva
            out_path = os.path.join(out_dir, f"sample_{bi}_{b}.png")
            overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
            cv2.imwrite(out_path, cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))
            print(f"[INFO] Saved {out_path}")