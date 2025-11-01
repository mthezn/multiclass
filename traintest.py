import os
import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from PIL import Image


def analyze_mask_classes(mask_path):
    """
    Analizza una maschera e ritorna le classi presenti
    """
    mask = Image.open(mask_path)
    mask = np.array(mask)

    # Trova tutte le classi uniche (escludi 0 = background)
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes > 0]

    return set(unique_classes.tolist())


def create_balanced_split(
        images_dir,
        masks_dir,
        output_dir,
        test_size=0.2,
        val_size=0.1,
        min_samples_per_class=5,
        random_state=42
):
    """
    Crea split train/val/test bilanciato rispetto alle classi presenti nelle maschere

    Args:
        images_dir: path alle immagini
        masks_dir: path alle maschere
        output_dir: dove salvare gli split
        test_size: % test set (default 20%)
        val_size: % validation set (default 10%)
        min_samples_per_class: minimo campioni per classe nel test/val
        random_state: seed per riproducibilità
    """

    print("🔍 Analisi dataset...")

    # 1. Raccogli info su tutte le immagini
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Mappa: image_file -> set di classi presenti
    image_to_classes = {}
    # Mappa: classe -> lista di immagini che la contengono
    class_to_images = defaultdict(list)

    for img_file in tqdm(image_files, desc="Analyzing masks"):
        mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(masks_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"⚠️ Warning: mask not found for {img_file}")
            continue

        classes = analyze_mask_classes(mask_path)
        image_to_classes[img_file] = classes

        for cls in classes:
            class_to_images[cls].append(img_file)

    # 2. Statistiche
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total images: {len(image_to_classes)}")
    print(f"  Classes found: {sorted(class_to_images.keys())}")
    print(f"\n  Images per class:")

    class_counts = {}
    for cls in sorted(class_to_images.keys()):
        count = len(class_to_images[cls])
        class_counts[cls] = count
        print(f"    Class {cls}: {count} images")

    # 3. Identifica classi rare (quelle con pochi campioni)
    rare_classes = [cls for cls, count in class_counts.items() if count < 50]
    print(f"\n⚠️ Rare classes (< 50 images): {rare_classes}")

    # 4. STRATEGIA DI SPLIT BILANCIATO
    # Per ogni classe rara, assicurati che train/val/test abbiano campioni

    train_images = set()
    val_images = set()
    test_images = set()

    np.random.seed(random_state)

    # Step 1: Assicura rappresentazione delle classi rare
    print(f"\n🎯 Ensuring rare class representation...")

    for rare_cls in rare_classes:
        images_with_rare = class_to_images[rare_cls].copy()
        np.random.shuffle(images_with_rare)

        n_images = len(images_with_rare)
        n_test = max(min_samples_per_class, int(n_images * test_size))
        n_val = max(min_samples_per_class, int(n_images * val_size))
        n_train = n_images - n_test - n_val

        # Assegna split
        test_images.update(images_with_rare[:n_test])
        val_images.update(images_with_rare[n_test:n_test + n_val])
        train_images.update(images_with_rare[n_test + n_val:])

        print(f"  Class {rare_cls}: {n_train} train, {n_val} val, {n_test} test")

    # Step 2: Assegna le immagini rimanenti mantenendo il bilanciamento
    remaining_images = set(image_to_classes.keys()) - train_images - val_images - test_images
    remaining_images = list(remaining_images)

    print(f"\n📦 Splitting remaining {len(remaining_images)} images...")

    # Stratified split basato sulla classe più rara nell'immagine
    stratify_labels = []
    for img in remaining_images:
        classes = image_to_classes[img]
        # Usa la classe più rara come label per stratificazione
        rarest_in_image = min(classes, key=lambda c: class_counts[c])
        stratify_labels.append(rarest_in_image)

    # Split stratificato
    if len(remaining_images) > 0:
        # Prima split: train vs (val+test)
        train_rem, temp_rem, _, temp_labels = train_test_split(
            remaining_images,
            stratify_labels,
            test_size=(test_size + val_size),
            random_state=random_state,
            stratify=stratify_labels
        )

        # Seconda split: val vs test
        if len(temp_rem) > 0:
            val_rem, test_rem = train_test_split(
                temp_rem,
                test_size=test_size / (test_size + val_size),
                random_state=random_state,
                stratify=temp_labels
            )

            train_images.update(train_rem)
            val_images.update(val_rem)
            test_images.update(test_rem)

    # 5. Verifica finale del bilanciamento
    print(f"\n✅ Final Split Statistics:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")

    print(f"\n📊 Class distribution per split:")

    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for split_name, split_images in splits.items():
        split_class_counts = defaultdict(int)
        for img in split_images:
            for cls in image_to_classes[img]:
                split_class_counts[cls] += 1

        print(f"\n  {split_name.upper()}:")
        for cls in sorted(split_class_counts.keys()):
            percentage = (split_class_counts[cls] / class_counts[cls]) * 100
            print(f"    Class {cls}: {split_class_counts[cls]:3d} images ({percentage:.1f}% of total)")

    # 6. Salva gli split
    print(f"\n💾 Saving splits to {output_dir}...")

    for split_name, split_images in splits.items():
        split_img_dir = os.path.join(output_dir, split_name, 'images')
        split_mask_dir = os.path.join(output_dir, split_name, 'masks')
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)

        for img_file in tqdm(split_images, desc=f"Copying {split_name}"):
            # Copia immagine
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(split_img_dir, img_file)
            shutil.copy2(src_img, dst_img)

            # Copia maschera
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            src_mask = os.path.join(masks_dir, mask_file)
            dst_mask = os.path.join(split_mask_dir, mask_file)
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)

    # 7. Salva file di testo con gli split
    for split_name, split_images in splits.items():
        split_file = os.path.join(output_dir, f'{split_name}_files.txt')
        with open(split_file, 'w') as f:
            for img in sorted(split_images):
                f.write(f"{img}\n")

    print(f"\n✅ Split completed successfully!")
    print(f"   Output directory: {output_dir}")

    return {
        'train': list(train_images),
        'val': list(val_images),
        'test': list(test_images),
        'class_counts': class_counts
    }


def verify_split_balance(output_dir):
    """
    Verifica che lo split sia bilanciato analizzando le maschere
    """
    print(f"\n🔍 Verifying split balance...")

    splits = ['train', 'val', 'test']

    for split_name in splits:
        masks_dir = os.path.join(output_dir, split_name, 'masks')

        if not os.path.exists(masks_dir):
            continue

        class_counts = defaultdict(int)
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]

        for mask_file in mask_files:
            mask_path = os.path.join(masks_dir, mask_file)
            classes = analyze_mask_classes(mask_path)
            for cls in classes:
                class_counts[cls] += 1

        print(f"\n{split_name.upper()} - Images per class:")
        for cls in sorted(class_counts.keys()):
            print(f"  Class {cls}: {class_counts[cls]}")


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Configurazione
    images_dir = "/path/to/your/images"
    masks_dir = "/path/to/your/masks"
    output_dir = "/path/to/output/split_dataset"

    # Crea split bilanciato
    split_info = create_balanced_split(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        test_size=0.2,  # 20% test
        val_size=0.1,  # 10% validation
        min_samples_per_class=5,  # Minimo 5 immagini per classe nel test/val
        random_state=42
    )

    # Verifica il bilanciamento
    verify_split_balance(output_dir)

    print("\n📁 Output structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/")
    print(f"    │   └── masks/")
    print(f"    ├── val/")
    print(f"    │   ├── images/")
    print(f"    │   └── masks/")
    print(f"    ├── test/")
    print(f"    │   ├── images/")
    print(f"    │   └── masks/")
    print(f"    ├── train_files.txt")
    print(f"    ├── val_files.txt")
    print(f"    └── test_files.txt")