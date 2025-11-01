import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import random
from copy import deepcopy


# ============================================================================
# APPROCCIO 1: CURRICULUM LEARNING - INIZIA CON CLASSI COMUNI
# ============================================================================

class CurriculumDataLoader:
    """Dataloader che introduce gradualmente le classi rare"""

    def __init__(self, original_dataloader, class_counts, total_epochs):
        self.original_dataloader = original_dataloader
        self.class_counts = class_counts
        self.total_epochs = total_epochs

        # Ordina classi per frequenza (più comuni prima)
        self.sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    def get_filtered_data(self, epoch):
        """Filtra i dati in base all'epoca corrente"""
        # Fase 1 (0-30%): Solo le 3 classi più comuni
        if epoch < self.total_epochs * 0.3:
            allowed_classes = [cls for cls, _ in self.sorted_classes[:3]]
        # Fase 2 (30-60%): Aggiungi classi intermedie
        elif epoch < self.total_epochs * 0.6:
            allowed_classes = [cls for cls, _ in self.sorted_classes[:6]]
        # Fase 3 (60%+): Tutte le classi con focus su quelle rare
        else:
            allowed_classes = [cls for cls, _ in self.sorted_classes]

        return allowed_classes


# ============================================================================
# APPROCCIO 2: SYNTHETIC DATA GENERATION PER CLASSI RARE
# ============================================================================

class SyntheticAugmentation:
    """Genera dati sintetici per classi rare"""

    def __init__(self, rare_threshold=50):
        self.rare_threshold = rare_threshold

    def create_synthetic_samples(self, rare_samples, num_synthetic=100):
        """Crea campioni sintetici da quelli esistenti"""
        synthetic_data = []

        for _ in range(num_synthetic):
            # Prendi due campioni casuali della stessa classe rara
            if len(rare_samples) >= 2:
                sample1, sample2 = random.sample(rare_samples, 2)

                # MixUp spaziale: combina ROI
                alpha = np.random.beta(0.4, 0.4)  # Parametri per mixup
                synthetic_roi = alpha * sample1['roi'] + (1 - alpha) * sample2['roi']

                # Augmentazioni aggressive
                synthetic_roi = self.apply_heavy_augmentation(synthetic_roi)

                synthetic_data.append({
                    'roi': synthetic_roi,
                    'label': sample1['label'],  # Stessa label
                    'is_synthetic': True
                })

        return synthetic_data

    def apply_heavy_augmentation(self, roi):
        """Augmentazioni molto aggressive per varietà"""
        import torchvision.transforms as T

        heavy_transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(45),  # Rotazione più forte
            T.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.7, 1.4),
                shear=(-20, 20)
            ),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 3.0))], p=0.5),
            T.ToTensor()
        ])

        return heavy_transform(roi.cpu()).to(roi.device)


# ============================================================================
# APPROCCIO 3: ENSEMBLE DI ESPERTI SPECIALIZZATI
# ============================================================================

class ExpertEnsemble(nn.Module):
    """Ensemble con esperti specializzati per diverse classi"""

    def __init__(self, num_classes, rare_classes):
        super().__init__()

        # Esperto generale (tutte le classi)
        self.general_expert = SurgicalToolClassifier(num_classes)

        # Esperto per classi rare
        self.rare_expert = SurgicalToolClassifier(len(rare_classes))

        # Esperto per classi comuni
        common_classes = num_classes - len(rare_classes)
        self.common_expert = SurgicalToolClassifier(common_classes)

        self.rare_classes = set(rare_classes)

        # Gating network per decidere quale esperto usare
        self.gating = nn.Sequential(
            nn.Linear(2048, 256),  # Feature dimension from EfficientNet
            nn.ReLU(),
            nn.Linear(256, 3),  # 3 esperti
            nn.Softmax(dim=1)
        )

    def forward(self, x, features=None):
        # Output da tutti gli esperti
        general_out = self.general_expert(x)
        rare_out = self.rare_expert(x)
        common_out = self.common_expert(x)

        # Se abbiamo features, usa gating network
        if features is not None:
            gate_weights = self.gating(features.mean(dim=(2, 3)))  # Global avg pool

            # Combina output pesati
            output = (gate_weights[:, 0:1] * general_out +
                      gate_weights[:, 1:2] * rare_out +
                      gate_weights[:, 2:3] * common_out)
        else:
            # Fallback: usa solo esperto generale
            output = general_out

        return output


# ============================================================================
# APPROCCIO 4: CONTRASTIVE LEARNING PER CLASSI RARE
# ============================================================================

class ContrastiveLoss(nn.Module):
    """Loss contrastiva per migliorare separazione classi rare"""

    def __init__(self, temperature=0.07, rare_classes=None):
        super().__init__()
        self.temperature = temperature
        self.rare_classes = set(rare_classes) if rare_classes else set()

    def forward(self, features, labels):
        # Normalizza features
        features = F.normalize(features, p=2, dim=1)

        # Calcola similarità
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Maschera per coppie positive (stessa classe)
        batch_size = labels.size(0)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Rimuovi diagonale (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)

        # Loss contrastiva con peso maggiore per classi rare
        weights = torch.ones_like(labels.squeeze())
        for i, label in enumerate(labels.squeeze()):
            if label.item() in self.rare_classes:
                weights[i] = 3.0  # Peso maggiore per classi rare

        # InfoNCE loss pesato
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = -(weights * mean_log_prob_pos).mean()

        return loss


# ============================================================================
# APPROCCIO 5: PROGRESSIVE SAMPLING
# ============================================================================

class ProgressiveSampler:
    """Sampler che aumenta gradualmente il focus sulle classi rare"""

    def __init__(self, class_counts, rare_threshold=50):
        self.class_counts = class_counts
        self.rare_classes = [cls for cls, count in class_counts.items()
                             if count < rare_threshold and cls != 0]
        self.common_classes = [cls for cls, count in class_counts.items()
                               if count >= rare_threshold and cls != 0]

    def get_sampling_weights(self, epoch, total_epochs):
        """Calcola pesi di sampling che evolvono nel tempo"""
        # Inizia con sampling uniforme
        progress = epoch / total_epochs

        weights = {}

        for cls, count in self.class_counts.items():
            if cls == 0:  # Skip background
                continue

            base_weight = 1.0 / count  # Inverse frequency

            if cls in self.rare_classes:
                # Aumenta peso per classi rare con il progresso
                rare_boost = 1.0 + progress * 4.0  # Fino a 5x peso
                weights[cls] = base_weight * rare_boost
            else:
                # Diminuisci leggermente peso per classi comuni
                common_reduction = 1.0 - progress * 0.5  # Fino a 0.5x peso
                weights[cls] = base_weight * common_reduction

        return weights


# ============================================================================
# APPROCCIO 6: META-LEARNING PER FEW-SHOT
# ============================================================================

class MAMLClassifier(nn.Module):
    """Meta-learning per adattamento rapido su classi rare"""

    def __init__(self, base_model, learning_rate=1e-3):
        super().__init__()
        self.base_model = base_model
        self.meta_lr = learning_rate

    def adapt(self, support_data, support_labels, adaptation_steps=5):
        """Adatta il modello su pochi esempi di una classe rara"""
        # Copia parametri per adattamento
        adapted_params = {}
        for name, param in self.base_model.named_parameters():
            adapted_params[name] = param.clone()

        # Gradient descent steps su support set
        for step in range(adaptation_steps):
            # Forward pass
            logits = self.forward_with_params(support_data, adapted_params)
            loss = F.cross_entropy(logits, support_labels)

            # Calcola gradienti
            grads = torch.autograd.grad(
                loss, adapted_params.values(),
                create_graph=True, retain_graph=True
            )

            # Update parametri
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.meta_lr * grad

        return adapted_params

    def forward_with_params(self, x, params):
        """Forward pass con parametri specifici"""
        # Implementa forward usando parametri custom
        # (semplificato per esempio)
        return self.base_model(x)


# ============================================================================
# TRAINING COMBINATO CON APPROCCI MULTIPLI
# ============================================================================

def train_with_multiple_strategies(
        model, classifier, dataloader, optimizer, device, epoch,
        class_counts, rare_threshold=50, total_epochs=100
):
    """Training che combina multiple strategie per classi rare"""

    model.eval()
    classifier.train()

    # Identifica classi rare
    rare_classes = [cls for cls, count in class_counts.items()
                    if count < rare_threshold and cls != 0]

    print(f"Epoca {epoch}: Classi rare = {rare_classes}")

    # 1. CURRICULUM LEARNING
    curriculum_loader = CurriculumDataLoader(dataloader, class_counts, total_epochs)
    allowed_classes = curriculum_loader.get_filtered_data(epoch)
    print(f"Classi attive: {allowed_classes}")

    # 2. PROGRESSIVE SAMPLING
    sampler = ProgressiveSampler(class_counts, rare_threshold)
    sampling_weights = sampler.get_sampling_weights(epoch, total_epochs)

    # 3. LOSS COMBINATION
    ce_loss = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss(rare_classes=rare_classes)

    # 4. SYNTHETIC DATA per classi rare (ogni 10 epoche)
    if epoch % 10 == 0 and epoch > 20:
        print("Generando dati sintetici per classi rare...")
        # Implementa generazione dati sintetici qui

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    rare_samples_seen = 0

    scaler = torch.amp.GradScaler()

    for i, (images, masks_gt) in enumerate(dataloader):
        # Stesso preprocessing di prima...
        images = images.to(device)
        masks_gt = masks_gt.to(device)

        with torch.no_grad():
            feat = model.image_encoder(images)
            pred_logits, out_dict = model.mask_decoder(feat)
            pred_logits = model.postprocess_masks(pred_logits, (1024, 1024), (1024, 1024))
            pred_masks = pred_logits > 0

        # Accumula istanze con filtering per curriculum
        instances_data = []
        for b in range(images.size(0)):
            inst_masks_list = extract_instances_optimized(pred_masks[b, 0])

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
                label = unique_labels[best_idx].item() - 1  # Convert to 0-based

                # CURRICULUM FILTERING
                if label not in allowed_classes:
                    continue

                roi = extract_roi_with_context(images[b], inst_mask,
                                               context_factor=0.3, target_size=224)

                # Applica peso di sampling
                sample_weight = sampling_weights.get(label, 1.0)

                instances_data.append({
                    'roi': roi,
                    'label': label,
                    'weight': sample_weight,
                    'is_rare': label in rare_classes
                })

                if label in rare_classes:
                    rare_samples_seen += 1

        if len(instances_data) == 0:
            continue

        # Weighted sampling basato su progressive weights
        weights = [inst['weight'] for inst in instances_data]
        if len(instances_data) > 32:  # Limita batch size
            indices = np.random.choice(
                len(instances_data), 32,
                replace=False, p=np.array(weights) / sum(weights)
            )
            selected_instances = [instances_data[i] for i in indices]
        else:
            selected_instances = instances_data

        # Costruisci batch
        batch_rois = torch.stack([inst['roi'] for inst in selected_instances])
        batch_labels = torch.tensor([inst['label'] for inst in selected_instances],
                                    dtype=torch.long, device=device)

        if batch_rois.max() > 1.0:
            batch_rois = batch_rois / 255.0

        # Forward pass con mixed loss
        with torch.amp.autocast(device_type='cuda'):
            logits = classifier(batch_rois)

            # Loss combination
            ce_loss_val = ce_loss(logits, batch_labels)

            # Contrastive loss su features (se disponibili)
            # features = classifier.get_features(batch_rois)  # Da implementare
            # contrastive_loss_val = contrastive_loss(features, batch_labels)

            # loss = ce_loss_val + 0.1 * contrastive_loss_val
            loss = ce_loss_val  # Semplificato per ora

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == batch_labels).sum().item()
        total_samples += len(batch_labels)

        if i % 50 == 0:
            current_acc = total_correct / max(1, total_samples)
            print(f"Batch {i}: Loss={loss.item():.4f}, Acc={current_acc:.3f}, "
                  f"Rare samples={rare_samples_seen}")

    avg_loss = total_loss / len(dataloader)
    acc = total_correct / max(1, total_samples)

    print(f"[Epoca {epoch}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}, "
          f"Campioni rari visti: {rare_samples_seen}")

    return {
        "loss": avg_loss,
        "acc": acc,
        "rare_samples": rare_samples_seen,
        "allowed_classes": len(allowed_classes)
    }