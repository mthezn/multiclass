import torch
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score,ConfusionMatrixDisplay
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import numpy as np
import time
from reportMulticlass import YOLOStyleReporter
import cv2

from datasets import load_dataset
from timm.utils import accuracy

from Dataset import ImageMaskDataset, CholecDataset, Kvasir, InstrumentDataset,InstrumentDatasetTest
import torch
from torch.utils.data import DataLoader

from instrumentClassifier import InstrumentClassifier,SurgicalToolClassifier
from utility import extract_instances, masked_global_avg_pool, get_instance_labels,extract_roi_with_context,extract_instances_optimized
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

    7: (255, 165, 0),      # Vessel_Sealer = arancione

}

def mask_to_rgb(mask, class_colors=CLASS_COLORS):
    """Converte una maschera di classi in immagine RGB con colori fissi"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in class_colors.items():
        rgb[mask == cls] = color
    return rgb

class_names = { #class_names per il print su immagine
    1: 'Large_Needle_Driver',
    2:'Forceps', #qui viene fatto in modo che tutte le calssi 2 e 3 vengano scrittte come froceps nell'immagine


    3: 'Grasping_Retractor',
    4: 'Maryland_Bipolar_Forceps' ,
     5: 'Monopolar_Curved_Scissors',
     6: 'Other',
     7 : 'Vessel_Sealer'
}

GLOBAL_CLASS_MAPPING = {#class names per il dataset che legge le cartelle e da un numero
    'Large_Needle_Driver': 1,
    'Prograsp_Forceps': 2,

    'Bipolar_Forceps': 2,#cambia a 2 se vuoi unificare le forceps
    'Grasping_Retractor': 3,
    'Maryland_Bipolar_Forceps': 4,
    'Monopolar_Curved_Scissors': 5,
    'Other': 6,
    'Vessel_Sealer': 7
}


image_dirs_test = ["MICCAImod/instrument_1_4_testing/instrument_dataset_1/left_frames",
                   "MICCAImod/instrument_1_4_testing/instrument_dataset_2/left_frames2",
                   "MICCAImod/instrument_1_4_testing/instrument_dataset_3/left_frames",
                   "MICCAImod/instrument_1_4_testing/instrument_dataset_4/left_frames",
                   "MICCAImod/instrument_5_8_testing/instrument_dataset_6/left_frames",
                   "MICCAImod/instrument_5_8_testing/instrument_dataset_7/left_frames",
                   "MICCAImod/instrument_5_8_testing/instrument_dataset_8/left_frames"]
mask_dirs_test = ["MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_1/gt/TypeSegmentationRescaled",
                  "MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_2/gt/TypeSegmentationRescaled",
                  "MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_3/gt/TypeSegmentationRescaled",
                  "MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_4/gt/TypeSegmentationRescaled",
                    "MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_6/gt/TypeSegmentationRescaled",
                  "MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_7/gt/TypeSegmentationRescaled",
                  "MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_8/gt/TypeSegmentationRescaled"]
image_dirs_train = [

    "MICCAI/instrument_1_4_training/instrument_dataset_1/left_frames",
]
mask_dirs_train = [
    "MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth"]

validation_transform = A.Compose([
    A.Resize(1024,1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

#print(len(datasetKvasir))


def contains_instrument(example):
    mask = np.array(example["color_mask"])  # o "segmentation" se diverso
    return np.any((mask == 169) | (mask == 170))

datasetTest = InstrumentDatasetTest(image_dirs=image_dirs_test, gt_dirs=mask_dirs_test, transform=validation_transform, class_to_id=GLOBAL_CLASS_MAPPING)
#datasetTest = CholecDataset(hf_dataset=filtered_ds, transform=validation_transform)
print(len(datasetTest))
#datasetTest = InstrumentDataset(image_dirs=image_dirs_train, gt_dirs=mask_dirs_train, transform=validation_transform, class_to_id=GLOBAL_CLASS_MAPPING)
dataloaderTest = DataLoader(datasetTest, batch_size=2, shuffle=True)




# CARICO UN MODELLO SAM
# sam_checkpoint = "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/checkpoints/sam_vit_b_01ec64.pth"
autosam_checkpoint = "/home/mdezen/distillation/checkpointsLight/autoSamFineUnetk57VL.pth"
model_type = "autoSamUnet"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = sam_model_registry[model_type](checkpoint=None, num_classes=1)
state_dict = torch.load(autosam_checkpoint, map_location=torch.device('cuda'))
model.load_state_dict(state_dict, strict=False)  # Load the state dict into the model
model.to(device=device)


model.eval()
classifier = SurgicalToolClassifier(num_classes=8, pretrained=True, dropout=0.3)
classifier_checkpoint = "checkpoints/01_09/classifierOM2Fk.pth"
state_dict = torch.load(classifier_checkpoint)
classifier.load_state_dict(state_dict)
classifier.to(device=device)
classifier.eval()
out_dir="results_classification_light"
max_batches=2


os.makedirs(out_dir, exist_ok=True)
model.eval()
classifier.eval()

colors = {}
rng = np.random.RandomState(42)
"""
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
            print(f"[INFO] Saved {out_path}")"""

model.eval()
classifier.eval()
total_correct = 0.0
total_samples = 0.0
timeDf = pd.DataFrame(columns=['time', 'index', 'pred', 'label', 'acc','f1'])
total_correct = 0
total_samples = 0
image_preds_all = []
image_labels_all = []
class_names_report = {
    1: 'Large_Needle_Driver',
    2: 'Forceps',
    3: 'Grasping_Retractor',
    4: 'Maryland_Bipolar_Forceps' ,
    5: 'Monopolar_Curved_Scissors',

     7 : 'Vessel_Sealer'
}
reporter = YOLOStyleReporter(
    class_names=class_names_report,
    save_dir=out_dir
)
i = 0

with torch.no_grad():
    for bi, (images, masks_gt) in enumerate(dataloaderTest):

        images = images.to(device)
        B, _, H, W = images.shape
        masks_gt = masks_gt.to(device)

        # 1. Segmentazione binaria
        feats = model.image_encoder(images)
        pred_logits, dict = model.mask_decoder(feats)
        pred_logits = model.postprocess_masks(pred_logits, (1024, 1024), (H, W))
        pred_masks = (pred_logits) > 0

        # Processa ogni immagine nel batch
        for b in range(images.size(0)):
            # ============ INIZIALIZZA PER QUESTA IMMAGINE ============
            image_preds = []  # Lista predizioni per questa immagine
            image_labels = []  # Lista label GT per questa immagine
            i = i + 1

            # Prepara immagine per visualizzazione
            img_np = ((images[b].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            img_np = np.ascontiguousarray(img_np)

            # Inizia timer per questa immagine
            start_time = time.time()

            # Estrai istanze di strumenti
            inst_masks_list = extract_instances_optimized(pred_masks[b, 0])
            overlay = img_np.copy().astype(np.float32)

            # ============ PROCESSA OGNI STRUMENTO NELL'IMMAGINE ============
            for inst_mask in inst_masks_list:
                inst_mask = inst_mask.to(device)

                # Trova label GT per questo strumento
                overlap = masks_gt[b] * inst_mask
                unique_labels, counts = torch.unique(overlap, return_counts=True)

                # Filtra solo label > 0 (ignora background)
                valid_mask = unique_labels > 0
                unique_labels = unique_labels[valid_mask]
                counts = counts[valid_mask]

                if len(unique_labels) == 0:
                    continue  # Skip se nessuna label valida

                # Prendi label con maggior overlap
                best_idx = torch.argmax(counts)
                label = unique_labels[best_idx].item()

                # Estrai ROI per classificazione
                roi = extract_roi_with_context(
                    images[b],
                    inst_mask,
                    context_factor=0.2,
                    target_size=224
                )
                roi = roi.unsqueeze(0)

                # Normalizza ROI se necessario
                if roi.max() > 1.0:
                    roi = roi / 255.0

                # Classificazione
                with torch.amp.autocast(device_type='cuda'):
                    logits = classifier(roi)
                pred = torch.argmax(logits, dim=1).item()
                """
                if pred == 3: 
                    pred = 2 #unifico le froceps alla stessa label 
                    """

                # ============ SALVA RISULTATI PER QUESTA IMMAGINE ============
                image_preds.append(pred)
                image_labels.append(label)

                # ============ VISUALIZZAZIONE ============
                mask_bool = inst_mask.cpu().numpy().astype(bool)
                color = np.array(CLASS_COLORS[pred], dtype=np.float32)

                # Applica colore alla mask
                overlay[mask_bool] = 0.5 * color + (1 - 0.5) * overlay[mask_bool]

                # Aggiungi testo al centro della mask
                ys, xs = np.where(mask_bool)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(
                        overlay,
                        class_names.get(pred, str(pred)),
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                # Debug print per ogni strumento
                print(f"Strumento: pred={pred}, label={label}")

            # ============ FINE ELABORAZIONE IMMAGINE ============
            end_time = time.time()
            processing_time =( end_time - start_time) *1000

            # Calcola accuracy per questa immagine
            if len(image_preds) > 0:
                correct_in_image = sum(1 for p, l in zip(image_preds, image_labels) if p == l)
                f1 = f1_score(image_labels, image_preds, average='macro', zero_division=0)
                image_labels_all.extend(image_labels)
                image_preds_all.extend(image_preds)

                reporter.update(
                    predictions=image_preds,
                    ground_truths=image_labels,
                    processing_time=processing_time
                )

                image_accuracy = correct_in_image / len(image_preds)

                # Aggiorna contatori globali
                total_correct += correct_in_image
                total_samples += len(image_preds)
            else:
                image_accuracy = 0
                correct_in_image = 0

            # ============ SALVA DATI IMMAGINE NEL DATAFRAME ============
            new_row = pd.DataFrame([{
                'time': processing_time,
                'index': f"{bi}_{b}",
                'pred': image_preds.copy(),  # Lista completa predizioni
                'label': image_labels.copy(),  # Lista completa label GT
                'acc': image_accuracy,
                'f1': f1,
                'num_instruments': len(image_preds)  # Numero strumenti trovati
            }])

            # Concatena al DataFrame principale
            timeDf = pd.concat([timeDf, new_row], ignore_index=True)

            # ============ SALVA IMMAGINE CON OVERLAY ============
            if i%10 == 0:
                out_path = os.path.join(out_dir, f"sample_{bi}_{b}.png")
                overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
                cv2.imwrite(out_path, cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))

            # ============ PRINT RISULTATI IMMAGINE ============
            print(f"\n[INFO] Image {bi}_{b}:")
            print(f"  - Instruments found: {len(image_preds)}")
            print(f"  - Predictions: {image_preds}")
            print(f"  - Ground truth: {image_labels}")
            print(f"  - Accuracy: {image_accuracy * 100:.1f}%")
            print(f"- f1score: {f1:.3f}")
            print(f"  - Processing time: {processing_time:.3f}ms")
            #print(f"  - Saved: {out_path}")

        # Print accuracy per batch
        batch_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"\n[BATCH {bi}] Running accuracy: {batch_accuracy * 100:.2f}%")

# ============ RISULTATI FINALI ============
print(f"\n" + "=" * 60)
print(f"FINAL RESULTS")
print(f"=" * 60)

overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
print(f"Total images processed: {len(timeDf)}")
print(f"Total instruments detected: {total_samples}")
print(f"Correct predictions: {total_correct}")
print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")

print(f"\nDataFrame shape: {timeDf.shape}")
print(f"Columns: {list(timeDf.columns)}")


print(f"\n" + "="*60)
print(f"DATAFRAME PREVIEW")
print(f"="*60)
print(timeDf.head())
cm = confusion_matrix(image_labels_all, image_preds_all, labels=list(CLASS_COLORS.keys())[1:])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASS_COLORS.keys())[1:])
disp.plot().figure_.savefig(os.path.join(out_dir, "confusion_matrix.png"))
reporter.generate_all_reports()

timeDf.to_csv(os.path.join(out_dir, "detection_results.csv"), index=False)
