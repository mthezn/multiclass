"""
Train and eval functions used in mainDecoupled.py, mainAuto.py
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import random
import torch.nn as nn
from torchvision.transforms import v2
from instrumentClassifier import InstrumentClassifier
from repvit_sam import SamPredictor
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from utility import calculate_iou, predict_points_boxes, get_bbox_centroids
from utility import extract_instances, masked_global_avg_pool, get_instance_labels, cutmix_with_mask, \
    extract_instances_optimized, extract_roi_with_context

import random
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import v2


class RareClassAugmentation:
    """Augmentazioni specifiche per classi rare"""

    @staticmethod
    def get_heavy_augmentation():
        """Augmentazioni aggressive per strumenti rari"""
        return transforms.Compose([
            transforms.RandomRotation(30),  # Più rotazione
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),  # Più spostamento
                scale=(0.8, 1.3),  # Più variazione scala
                shear=(-15, 15)  # Aggiunge shear
            ),
            transforms.ColorJitter(
                brightness=0.4,  # Più variazione colore
                contrast=0.4,
                saturation=0.3,
                hue=0.15
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),

        ])

    @staticmethod
    def apply_mixup_for_rare_classes(images, labels, rare_classes, mixup_alpha=0.4):
        """MixUp specifico per classi rare"""
        batch_size = images.size(0)
        rare_indices = [i for i, label in enumerate(labels) if label.item() in rare_classes]

        if len(rare_indices) < 2:
            return images, labels

        # Crea coppie di immagini rare per mixup
        mixed_images = images.clone()
        mixed_labels = labels.clone()

        for i in rare_indices[::2]:  # Prendi ogni seconda immagine rara
            if i + 1 < len(rare_indices):
                j = rare_indices[i + 1]
                lam = np.random.beta(mixup_alpha, mixup_alpha)

                mixed_images[i] = lam * images[i] + (1 - lam) * images[j]
                # Soft label per mixup
                mixed_labels[i] = lam
                print(mixed_labels)

        return mixed_images, mixed_labels


class ImbalanceHandler:
    """Gestisce il problema delle classi sbilanciate"""

    def __init__(self, class_counts, strategy='focal_loss'):
        self.class_counts = class_counts
        self.strategy = strategy
        self.total_samples = sum(class_counts.values())
        self.num_classes = 9

    def get_class_weights(self):
        """Calcola pesi per bilanciare le classi (ignora background/classe 0)"""
        weights = {}

        # Filtra classe 0 (background) e ricalcola total e num_classes
        valid_classes = {k: v for k, v in self.class_counts.items() if k != 0}
        valid_total_samples = sum(valid_classes.values())
        valid_num_classes = len(valid_classes)

        for class_idx, count in valid_classes.items():
            # Inverse frequency weighting solo per classi valide
            weights[class_idx] = valid_total_samples / (valid_num_classes * count)

        # Normalizza i pesi
        if weights:  # Check se ci sono pesi validi
            max_weight = max(weights.values())
            weights = {k: v / max_weight for k, v in weights.items()}

        # Converti a tensor (dimensione = num_classes originale)
        weight_tensor = torch.ones(self.num_classes)  # Default weight = 1

        # Imposta peso 0 per background (classe 0)
        weight_tensor[0] = 0.0

        # Assegna pesi calcolati alle classi valide
        for class_idx, weight in weights.items():
            if class_idx < self.num_classes:  # Safety check
                weight_tensor[class_idx] = weight
        print(weight_tensor)
        return weight_tensor

    def create_sampler(self, dataset_labels):
        """Crea sampler per bilanciare durante training"""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in dataset_labels]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset_labels),
            replacement=True
        )


class FocalLoss(nn.Module):
    """Focal Loss per classi sbilanciate"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_one_epoch_instruments_efficientnet(
        model, classifier, dataloader, optimizer, device, run, epoch, criterion, class_counts,
        loss_scaler=None, use_mixup=True, focal_loss=False, roi_size=224
):
    model.eval()  # Segmentatore in eval
    classifier.train()  # EfficientNet in train
    rare_classes = [cls for cls, count in class_counts.items() if count < 200]
    total_loss = 0.0
    total_correct = 0
    class_5_total = 0
    total_samples = 0
    heavy_augment = RareClassAugmentation.get_heavy_augmentation()
    imbalance_handler = ImbalanceHandler(class_counts)
    class_weights = imbalance_handler.get_class_weights().to(device)
    if epoch < 3:
        allowed_classes = [1, 2, 3, 4]  # Le tue classi più comuni
    elif epoch < 6:
        allowed_classes = [1, 2, 3, 4, 5, 6, 7]
    elif epoch < 8:
        allowed_classes = [1, 2, 3, 4, 5, 6, 8]
    else:
        allowed_classes = list(range(1, 10))
    # MixUp/CutMix
    if use_mixup:
        cutmix = v2.CutMix(num_classes=8, alpha=1.0)
        mixup = v2.MixUp(num_classes=8, alpha=0.2)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    # Focal Loss
    if focal_loss:
        focal_loss = FocalLoss(alpha=class_weights, gamma=3.0)
        criterion = focal_loss

    scaler = torch.amp.GradScaler()
    bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")

    # Gradient accumulation
    accumulation_steps = 2  # Riduci se hai memoria
    rare_samples = 0
    rare_buffer = []  # Salva ROI della classe più rara per riutilizzo
    max_buffer_size = 100
    for i, (images, masks_gt) in bar:
        images = images.to(device, non_blocking=True)
        masks_gt = masks_gt.to(device, non_blocking=True)

        # Forward del segmentatore (frozen)
        with torch.no_grad():
            feat = model.image_encoder(images)
            pred_logits, out_dict = model.mask_decoder(feat)
            pred_logits = model.postprocess_masks(pred_logits, (1024, 1024), (1024, 1024))
            pred_masks = pred_logits > 0  # (B, 16, H, W)

        # Estrai ROI per EfficientNet
        batch_rois = []
        batch_labels = []

        for b in range(images.size(0)):
            # Assumiamo che usi solo il primo canale della predizione
            inst_masks_list = extract_instances_optimized(pred_masks[b, 0], min_area=300)

            for inst_mask in inst_masks_list:
                inst_mask = inst_mask.to(device)

                # Trova label GT corrispondente
                overlap = masks_gt[b] * inst_mask
                unique_labels, counts = torch.unique(overlap, return_counts=True)
                valid_mask = unique_labels > 0
                unique_labels = unique_labels[valid_mask]
                counts = counts[valid_mask]

                if len(unique_labels) == 0:
                    continue

                best_idx = torch.argmax(counts)
                label = unique_labels[best_idx].item()
                if label not in allowed_classes:
                    continue

                # Estrai ROI con contesto
                roi = extract_roi_with_context(images[b], inst_mask,
                                               context_factor=0.5,
                                               target_size=roi_size)
                """if label == 5:
                    # Crea multiple copie con augmentation diverse
                    n_copies = max(3, int(270 / 75))  # Calcola quante copie servono

                    for copy_idx in range(n_copies):
                        roi_pil = transforms.ToPILImage()(roi.cpu())
                        roi_aug = transforms.ToTensor()(heavy_augment(roi_pil)).to(device)
                        batch_rois.append(roi_aug)
                        batch_labels.append(label)

                    # Salva nel buffer per riutilizzo
                    if len(rare_buffer) < max_buffer_size:
                        rare_buffer.append((roi.clone(), label))
                elif label in rare_classes:
                    n_copies = 2  # Duplica anche le altre rare
                    for _ in range(n_copies):
                        roi_pil = transforms.ToPILImage()(roi.cpu())
                        roi_aug = transforms.ToTensor()(heavy_augment(roi_pil)).to(device)
                        batch_rois.append(roi_aug)
                        batch_labels.append(label)
                    rare_samples += n_copies
                # rarest_samples += n_copies
                else:"""
                if label in rare_classes:
                    # ===== STRATEGIA 1: Augmentation per classe 5 =====
                    # Aggiungi SEMPRE il campione originale
                    batch_rois.append(roi)
                    batch_labels.append(label)
                    rare_samples += 1

                    # Poi aggiungi una copia augmentata
                    roi_pil = transforms.ToPILImage()(roi.cpu())
                    roi_aug = transforms.ToTensor()(heavy_augment(roi_pil)).to(device)
                    batch_rois.append(roi_aug)
                    batch_labels.append(label)
                    rare_samples += 1

                    # Salva nel buffer per riutilizzo
                    if len(rare_buffer) < max_buffer_size:
                        rare_buffer.append((roi.clone(), label))
                else:
                    # Classi comuni: aggiungi normalmente
                    batch_rois.append(roi)
                    batch_labels.append(label)

            # ===== STRATEGIA 2: Augmentation per altre classi rare =====

        if len(batch_rois) == 0:
                continue

    # Costruisci batch tensor
        batch_rois = torch.stack(batch_rois, dim=0)  # (N, 3, roi_size, roi_size)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)

        trigger = np.random.random()
        # MixUp/CutMix (opzionale)
        if use_mixup and trigger > 0.5:
            batch_rois, batch_labels = cutmix_or_mixup(batch_rois, batch_labels)

        # batch_rois, batch_labels = RareClassAugmentation.apply_mixup_for_rare_classes(
        #    batch_rois, batch_labels, rare_classes
        # )

        # ROI siano in range [0,1]
        if batch_rois.max() > 1.0:
            batch_rois = batch_rois / 255.0

        # Forward pass EfficientNet
        with torch.amp.autocast(device_type='cuda'):
            logits = classifier(batch_rois)
            if use_mixup and trigger > 0.5:
                batch_labels = torch.argmax(batch_labels, dim=1)
            loss = criterion(logits, batch_labels) / accumulation_steps
        preds = torch.argmax(logits, dim=1)
        class_5_correct = ((preds == batch_labels) & (5 in batch_labels)).sum().item()  # 4 se 0-indexed
        if 5 in batch_labels:
            class_5_total = (batch_labels == 5).sum().item()

        # Log specifico
        if class_5_total > 0:
            class_5_acc = class_5_correct / class_5_total
            if run is not None:
                run.log({"class_5_accuracy": class_5_acc})
        # Backward con accumulation
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping per stabilità
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Metrics
        total_loss += loss.item() * accumulation_steps
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)

            if isinstance(batch_labels, torch.Tensor) and batch_labels.dim() > 1:
                # Per mixup/cutmix
                batch_labels_hard = torch.argmax(batch_labels, dim=1)
            else:
                batch_labels_hard = batch_labels

            total_correct += (preds == batch_labels_hard).sum().item()
            total_samples += batch_labels_hard.size(0)

    # Logging e progress
        if i % 50 == 0:
            current_acc = total_correct / max(1, total_samples)
            current_loss = total_loss / max(1, i + 1)
            bar.set_description(f"Loss: {current_loss:.4f}, Acc: {current_acc:.3f}, Samples: {total_samples}")
            print(rare_samples)

            if run is not None:
                run.log({
                    "train_batch_loss": loss.item() * accumulation_steps,
                    "train_batch_acc": current_acc,
                    "epoch": epoch,
                    "batch": i,
                    "samples_processed": total_samples
                })

        # Memory cleanup
        del batch_rois, batch_labels, logits
        torch.cuda.empty_cache()


        # Final metrics
    avg_loss = total_loss / max(1, len(dataloader))
    acc = total_correct / max(1, total_samples)

    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Total samples: {total_samples}")

    if run is not None:
        run.log({
            "train/loss": avg_loss,
            "train/acc": acc,
            "train/samples": total_samples,
            "epoch": epoch
        })

    return {"loss": avg_loss, "acc": acc, "samples": total_samples}

"""def train_one_epoch_instruments_efficientnet(
        model, classifier, dataloader, optimizer, device, run, epoch, criterion, class_counts,
        loss_scaler=None, use_mixup=True, focal_loss=False, roi_size=224
):
    model.eval()
    classifier.train()

    rarest_class = 5
    rare_classes = [cls for cls, count in class_counts.items() if count < 200]

    print(f"🎯 Epoch {epoch} - Focus su classe {rarest_class}")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class5_correct = 0
    class5_total = 0
    class5_generated = 0

    # ===== AUGMENTATION PROGRESSIVE: meno aggressiva con l'andare delle epoche =====
    # Inizio: augmentation pesante per generare varietà
    # Fine: augmentation leggera per non distorcere troppo la distribuzione reale

    if epoch < 3:
        aug_intensity = "heavy"
        class5_multiplier = 6
    elif epoch < 6:
        aug_intensity = "medium"
        class5_multiplier = 5
    else:
        aug_intensity = "light"
        class5_multiplier = 4

    # Augmentation PESANTE (prime epoche)
    heavy_augment_pil = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomVerticalFlip(p=0.7),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    # Augmentation MEDIA
    medium_augment_pil = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomVerticalFlip(p=0.6),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    ])

    # Augmentation LEGGERA (epoche finali - simile a validation)
    light_augment_pil = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ])

    # Seleziona augmentation basata sull'epoca
    if aug_intensity == "heavy":
        augment_pil = heavy_augment_pil
    elif aug_intensity == "medium":
        augment_pil = medium_augment_pil
    else:
        augment_pil = light_augment_pil

    # Augmentation tensor (sempre leggera)


    # Curriculum learning
    if epoch < 3:
        allowed_classes = [1, 2, 3, 4, 5]
    elif epoch < 6:
        allowed_classes = [1, 2, 3, 4, 5, 6, 7]
    elif epoch < 8:
        allowed_classes = [1, 2, 3, 4, 5, 6, 8]
    else:
        allowed_classes = list(range(1, 10))

    # Loss con peso DECRESCENTE per classe 5
    # Inizio: peso alto per forzare apprendimento
    # Fine: peso normale per evitare overfitting
    if epoch < 3:
        class5_weight = 10.0
    elif epoch < 6:
        class5_weight = 6.0
    else:
        class5_weight = 3.0

    class_weights = torch.ones(9, device=device)
    class_weights[4] = class5_weight
    criterion_weighted = torch.nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler()
    bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")

    accumulation_steps = 2
    class5_buffer = []
    max_buffer_size = 150

    for i, (images, masks_gt) in bar:
        images = images.to(device, non_blocking=True)
        masks_gt = masks_gt.to(device, non_blocking=True)

        with torch.no_grad():
            feat = model.image_encoder(images)
            pred_logits, out_dict = model.mask_decoder(feat)
            pred_logits = model.postprocess_masks(pred_logits, (1024, 1024), (1024, 1024))
            pred_masks = pred_logits > 0

        batch_rois = []
        batch_labels = []

        for b in range(images.size(0)):
            inst_masks_list = extract_instances_optimized(pred_masks[b, 0], min_area=300)

            for inst_mask in inst_masks_list:
                inst_mask = inst_mask.to(device)

                overlap = masks_gt[b] * inst_mask
                unique_labels, counts = torch.unique(overlap, return_counts=True)
                valid_mask = unique_labels > 0
                unique_labels = unique_labels[valid_mask]
                counts = counts[valid_mask]

                if len(unique_labels) == 0:
                    continue

                best_idx = torch.argmax(counts)
                label = unique_labels[best_idx].item()

                if label not in allowed_classes:
                    continue

                roi = extract_roi_with_context(images[b], inst_mask,
                                               context_factor=0.5,
                                               target_size=roi_size)

                # ===== STRATEGIA BILANCIATA PER CLASSE 5 =====
                if label == 5:
                    # IMPORTANTE: Aggiungi SEMPRE il campione ORIGINALE (senza augmentation)
                    # Questo aiuta il modello a generalizzare
                    batch_rois.append(roi)
                    batch_labels.append(label)
                    class5_generated += 1

                    # Poi aggiungi copie augmentate (ma meno dell'originale)
                    n_augmented = class5_multiplier - 1  # -1 perché abbiamo già l'originale

                    for copy_idx in range(n_augmented):
                        roi_pil = transforms.ToPILImage()(roi.cpu())

                        # Mix di augmentation: 50% heavy, 50% light
                        # Così il modello vede sia versioni distorte che pulite
                        if random.random() < 0.5 and aug_intensity != "light":
                            roi_aug = augment_pil(roi_pil)
                        else:
                            roi_aug = light_augment_pil(roi_pil)

                        roi_aug = transforms.ToTensor()(roi_aug).to(device)

                        # RandomErasing solo 30% delle volte

                        batch_rois.append(roi_aug)
                        batch_labels.append(label)
                        class5_generated += 1

                    # Salva nel buffer
                    if len(class5_buffer) < max_buffer_size:
                        class5_buffer.append(roi.clone())

                elif label in rare_classes:
                    # Altre classi rare: sempre includi originale + 1 augmentato
                    batch_rois.append(roi)
                    batch_labels.append(label)

                    roi_pil = transforms.ToPILImage()(roi.cpu())
                    roi_aug = light_augment_pil(roi_pil)  # Solo augmentation leggera
                    roi_aug = transforms.ToTensor()(roi_aug).to(device)
                    batch_rois.append(roi_aug)
                    batch_labels.append(label)
                else:
                    # Classi comuni: solo originale
                    batch_rois.append(roi)
                    batch_labels.append(label)

        # ===== INJETTA DAL BUFFER CON PROBABILITÀ DECRESCENTE =====
        # Prime epoche: injetta molto
        # Epoche finali: injetta poco per non distorcere distribuzione
        if epoch < 4:
            inject_prob = 0.8
            n_inject = 6
        elif epoch < 7:
            inject_prob = 0.5
            n_inject = 4
        else:
            inject_prob = 0.3
            n_inject = 2

        if len(class5_buffer) > 0 and random.random() < inject_prob:
            n_inject = min(len(class5_buffer), n_inject)
            for _ in range(n_inject):
                roi = random.choice(class5_buffer)

                # 50% originale, 50% augmentato
                if random.random() < 0.5:
                    batch_rois.append(roi)
                else:
                    roi_pil = transforms.ToPILImage()(roi.cpu())
                    roi_aug = light_augment_pil(roi_pil)
                    roi_aug = transforms.ToTensor()(roi_aug).to(device)
                    batch_rois.append(roi_aug)

                batch_labels.append(5)
                class5_generated += 1

        if len(batch_rois) == 0:
            continue

        batch_rois = torch.stack(batch_rois, dim=0)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)

        # ===== MIXUP RIDOTTO =====
        # Solo 30% delle volte e solo se ci sono abbastanza campioni classe 5
        if use_mixup and 5 in batch_labels and random.random() < 0.3:
            class5_count = (batch_labels == 5).sum()
            if class5_count >= 4:  # Solo se ci sono almeno 4 campioni
                cutmix = v2.CutMix(num_classes=9, alpha=0.5)  # Alpha ridotto
                mixup = v2.MixUp(num_classes=9, alpha=0.2)
                cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
                batch_rois, batch_labels = cutmix_or_mixup(batch_rois, batch_labels)

        # Normalizza
        if batch_rois.max() > 1.0:
            batch_rois = batch_rois / 255.0

        # Forward pass
        with torch.amp.autocast(device_type='cuda'):
            logits = classifier(batch_rois)

            # Handle mixup labels
            if isinstance(batch_labels, torch.Tensor) and batch_labels.dim() > 1:
                loss = criterion_weighted(logits, batch_labels) / accumulation_steps
            else:
                loss = criterion_weighted(logits, batch_labels) / accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Metrics
        total_loss += loss.item() * accumulation_steps
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)

            if isinstance(batch_labels, torch.Tensor) and batch_labels.dim() > 1:
                batch_labels_hard = torch.argmax(batch_labels, dim=1)
            else:
                batch_labels_hard = batch_labels

            total_correct += (preds == batch_labels_hard).sum().item()
            total_samples += batch_labels_hard.size(0)

            class5_mask = batch_labels_hard == 5
            if class5_mask.sum() > 0:
                class5_preds = preds[class5_mask]
                class5_labels = batch_labels_hard[class5_mask]
                class5_correct += (class5_preds == class5_labels).sum().item()
                class5_total += class5_mask.sum().item()

        # Logging
        if i % 25 == 0:
            current_acc = total_correct / max(1, total_samples)
            current_loss = total_loss / max(1, i + 1)
            class5_acc = class5_correct / max(1, class5_total)

            bar.set_description(
                f"Loss: {current_loss:.4f} | Acc: {current_acc:.3f} | "
                f"C5: {class5_acc:.3f} ({class5_correct}/{class5_total}) | Aug: {aug_intensity}"
            )

            if run is not None:
                run.log({
                    "train_batch_loss": loss.item() * accumulation_steps,
                    "train_batch_acc": current_acc,
                    "class5_batch_acc": class5_acc,
                    "class5_weight": class5_weight,
                    "aug_intensity": 2 if aug_intensity == "heavy" else (1 if aug_intensity == "medium" else 0),
                    "epoch": epoch,
                    "batch": i,
                })

        del batch_rois, batch_labels, logits
        torch.cuda.empty_cache()

    # Final metrics
    avg_loss = total_loss / max(1, len(dataloader))
    acc = total_correct / max(1, total_samples)
    class5_final_acc = class5_correct / max(1, class5_total)

    print(f"\n{'=' * 60}")
    print(f"[Epoch {epoch}] Aug: {aug_intensity} | Weight: {class5_weight:.1f}")
    print(f"  Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    print(f"  🎯 CLASSE 5: {class5_final_acc:.4f} ({class5_correct}/{class5_total})")
    print(f"  Generated: {class5_generated} samples")
    print(f"{'=' * 60}\n")

    if run is not None:
        run.log({
            "train/loss": avg_loss,
            "train/acc": acc,
            "train/class5_acc": class5_final_acc,
            "train/class5_total": class5_total,
            "train/class5_generated": class5_generated,
            "epoch": epoch
        })

    return {
        "loss": avg_loss,
        "acc": acc,
        "class5_acc": class5_final_acc,
        "class5_samples": class5_total
    }

def train_one_epoch_instruments(
        model, classifier, dataloader, optimizer, device, run, epoch, criterion, loss_scaler=None
):
    model.eval()
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    cutmix = v2.CutMix(num_classes=9)
    mixup = v2.MixUp(num_classes=9)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    scaler = torch.amp.GradScaler()
    bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")

    for i, (images, masks_gt) in bar:
        images = images.to(device)  # (B,3,H,W)
        masks_gt = masks_gt.to(device)  # (B,H,W)

        with torch.no_grad():
            feat = model.image_encoder(images)
            pred_logits, out_dict = model.mask_decoder(feat)

            pred_logits = model.postprocess_masks(pred_logits, (1024, 1024), (1024, 1024))

            # print(pred_logits[0:1].shape)

            pred_masks = pred_logits > 0  # (B,16,H,W)
            feat_out = out_dict["feat"]
            # print(feat_out.shape)
        # accumula istanze
        inst_imgs, inst_masks, inst_labels, inst_feats = [], [], [], []
        for b in range(images.size(0)):
            inst_masks_list = extract_instances(pred_masks[b, 0])  # lista di HxW
            for inst_mask in inst_masks_list:
                inst_mask = inst_mask.to(device)  # (H,W)

                # etichetta GT corrispondente
                unique_labels = torch.unique(masks_gt[b] * inst_mask)
                unique_labels = unique_labels[unique_labels > 0]
                if len(unique_labels) == 0:
                    continue

                label = unique_labels[0].item()
                inst_imgs.append(images[b].unsqueeze(0))

                inst_feats.append(feat_out[b:b + 1])  # (N_inst,C,H_Feat,W_feat)
                inst_masks.append(inst_mask.unsqueeze(0).unsqueeze(0))  # (1,1,H,W)
                inst_labels.append(label)

        if len(inst_imgs) == 0:
            continue

        # concatena
        batch_imgs = torch.cat(inst_imgs, dim=0).to(device)  # (N_inst,3,H,W)
        batch_masks = torch.cat(inst_masks, dim=0).to(device)  # (N_inst,1,H,W)
        batch_labels = torch.tensor(inst_labels, dtype=torch.long, device=device)
        # print(batch_labels)

        # print(batch_labels)
        # batch_imgs,batch_masks, batch_labels = cutmix_with_mask(batch_imgs, batch_masks, batch_labels, alpha=1.0)
        # cv2.imwrite(f"debug_samples/batch_img_{i}.png", ((batch_imgs[0].permute(1,2,0).cpu().numpy()*0.5 +0.5)*255).astype(np.uint8))
        # print(batch_labels)
        batch_feats = torch.cat(inst_feats, dim=0).to(device)  # (N_inst,C,H_feat,W_feat)

        # classificazione con masked pooling
        logits = classifier(batch_imgs, batch_masks)  # (N_inst,num_classes)
        loss = criterion(logits, batch_labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        # print(preds)
        # print(batch_labels)
        # targets = torch.argmax(batch_labels, dim=1)
        # print(targets)
        total_correct += ((preds == batch_labels).sum().item())
        total_samples += batch_labels.size(0)

        avg_loss_batch = total_loss / max(1, total_samples)
        bar.set_description(f"Loss: {loss.item():.4f}")
        run.log({"train_loss": avg_loss_batch, "epoch": epoch + 1, "batch": i + 1})

        del batch_imgs, batch_masks, batch_labels, logits, loss, batch_feats
        torch.cuda.empty_cache()
        del inst_imgs, inst_masks, inst_labels

    avg_loss = total_loss / max(1, len(dataloader))
    acc = total_correct / max(1, total_samples)

    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    if run is not None:
        run.log({"train/loss": avg_loss, "train/acc": acc, "epoch": epoch})

    return {"loss": avg_loss, "acc": acc}


@torch.no_grad()
def validate_one_epoch_instruments(model, classifier, dataloader, device, run, epoch, criterion):
    model.eval()
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, masks_gt in dataloader:
        images = images.to(device)
        masks_gt = masks_gt.to(device)

        # step 1: segmentazione binaria dal modello
        feats = model.image_encoder(images)
        pred_logits, out_dict = model.mask_decoder(feats)
        pred_logits = model.postprocess_masks(pred_logits, (1024, 1024), (1024, 1024))
        pred_masks = pred_logits > 0
        feat_out = out_dict["feat"]

        inst_imgs, inst_masks, inst_labels, inst_feats = [], [], [], []
        for b in range(images.size(0)):
            inst_masks_list = extract_instances(pred_masks[b, 0])
            for inst_mask in inst_masks_list:
                inst_mask = inst_mask.to(device)

                # recupera label GT
                unique_labels = torch.unique(masks_gt[b] * inst_mask)
                unique_labels = unique_labels[unique_labels > 0]
                if len(unique_labels) == 0:
                    # print("CIAO")
                    continue

                label = unique_labels[0].item()
                inst_imgs.append(images[b].unsqueeze(0))  # (1,3,H,W)
                inst_masks.append(inst_mask.unsqueeze(0).unsqueeze(0))
                inst_feats.append(feat_out[b:b + 1])  # (1,C,H_feat,W_feat)
                inst_labels.append(label)

                # concatena
        batch_imgs = torch.cat(inst_imgs, dim=0).to(device)  # (N_inst,3,H,W)
        batch_masks = torch.cat(inst_masks, dim=0).to(device)  # (N_inst,1,H,W)
        batch_labels = torch.tensor(inst_labels, dtype=torch.long, device=device)
        batch_feats = torch.cat(inst_feats, dim=0).to(device)  # (N_inst,C,H_feat,W_feat)
        # classificazione con masked pooling
        logits = classifier(batch_imgs, batch_masks)  # (N_inst,num_classes)
        loss = criterion(logits, batch_labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

        del batch_imgs, batch_masks, batch_labels, logits, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, len(dataloader))
    acc = total_correct / max(1, total_samples)

    print(f"[Val Epoch {epoch}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    if run is not None:
        run.log({"val/loss": avg_loss, "val/acc": acc, "epoch": epoch})

    return avg_loss
"""


def validate_efficientnet(model, classifier, val_dataloader, device, criterion, epoch, run, roi_size=224):
    """Validation per EfficientNet"""
    model.eval()
    classifier.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        bar = tqdm(val_dataloader, desc="Validation")

        for images, masks_gt in bar:
            images = images.to(device, non_blocking=True)
            masks_gt = masks_gt.to(device, non_blocking=True)

            # Segmentation
            feat = model.image_encoder(images)
            pred_logits, out_dict = model.mask_decoder(feat)
            pred_logits = model.postprocess_masks(pred_logits, (1024, 1024), (1024, 1024))
            pred_masks = pred_logits > 0

            # Extract ROIs
            batch_rois = []
            batch_labels = []

            for b in range(images.size(0)):
                inst_masks_list = extract_instances_optimized(pred_masks[b, 0])

                for inst_mask in inst_masks_list:
                    inst_mask = inst_mask.to(device)
                    overlap = masks_gt[b] * inst_mask
                    unique_labels, counts = torch.unique(overlap, return_counts=True)

                    # Filtra solo label > 0
                    valid_mask = unique_labels > 0
                    unique_labels = unique_labels[valid_mask]
                    counts = counts[valid_mask]

                    if len(unique_labels) == 0:
                        continue
                    best_idx = torch.argmax(counts)
                    label = unique_labels[best_idx].item()
                    roi = extract_roi_with_context(images[b], inst_mask,
                                                   context_factor=0.2,  # Meno context in val
                                                   target_size=roi_size)

                    batch_rois.append(roi)
                    batch_labels.append(label)

            if len(batch_rois) == 0:
                continue

            # Forward
            batch_rois = torch.stack(batch_rois, dim=0)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)

            # Normalize to [0,1] per il classificatore
            if batch_rois.max() > 1.0:
                batch_rois = batch_rois / 255.0

            with torch.amp.autocast(device_type='cuda'):
                logits = classifier(batch_rois)
                loss = criterion(logits, batch_labels)

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            current_acc = total_correct / max(1, total_samples)
            bar.set_description(f"Val Acc: {current_acc:.3f}")

    avg_loss = total_loss / max(1, len(val_dataloader))
    acc = total_correct / max(1, total_samples)
    if run is not None:
        run.log({"val/loss": avg_loss, "val/acc": acc, "epoch": epoch})

    print(f"Validation: Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Samples: {total_samples}")

    return {"loss": avg_loss, "acc": acc, "samples": total_samples}
