import copy
import albumentations as A
from torch.utils.data import DataLoader
from wandb.integration.sklearn.plot.classifier import classifier

from Dataset import CholecDataset
from Dataset import InstrumentDataset
from instrumentClassifier import InstrumentClassifier
from modeling.build_sam import sam_model_registry
from Dataset import ImageMaskDataset
from collections import Counter

from utils import *
from albumentations.pytorch import ToTensorV2
import wandb
import numpy as np
import torch.nn as nn
from engine import train_one_epoch_fine, validate_one_epoch_fine, train_one_epoch_instruments, \
    validate_one_epoch_instruments
from torch.utils.data import ConcatDataset

from timm.optim import create_optimizer_v2
from timm.utils import NativeScaler
import torch
import gc
import torch.nn.functional as F
from datasets import load_dataset
from utility import generate_random_name, contains_instrument

############################################################################################################


wandb.login(key='14497a5de45116d579bde37168ccf06f78c2928e')  # Replace 'your_api_key' with your actual API key
name = "classifier"+generate_random_name(5)

#datasetCholec = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)



#filtered_ds = datasetCholec["train"].filter()


seed = 42
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

## dataset
batch_size = 1
train_transform = A.Compose([
    A.Resize(1024, 1024),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    #A.RandomCrop(height=500, width=500, p=0.5),
    #A.PadIfNeeded(min_height=500, min_width=500,
     #                    border_mode=0, fill=0, fill_mask=0),

    #A.ColorJitter(p=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    A.Resize(height=1024, width=1024),
    ToTensorV2()
])


validation_transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
#DIRECTORIES
image_dirs_val = ["MICCAI/instrument_1_4_training/instrument_dataset_3/left_frames"]
mask_dirs_val = ["MICCAI/instrument_1_4_training/instrument_dataset_3/ground_truth"
                ]

image_dirs_train = [
    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/test",
    "MICCAI/instrument_1_4_training/instrument_dataset_1/left_frames",
    "MICCAI/instrument_1_4_training/instrument_dataset_2/left_frames",
    "MICCAI/instrument_1_4_training/instrument_dataset_4/left_frames",
    "MICCAI/instrument_5_8_training/instrument_dataset_5/left_frames",
    "MICCAI/instrument_5_8_training/instrument_dataset_6/left_frames",
    "MICCAI/instrument_5_8_training/instrument_dataset_7/left_frames",
    "MICCAI/instrument_5_8_training/instrument_dataset_8/left_frames",
]
mask_dirs_train = [
    "MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth",

    "MICCAI/instrument_1_4_training/instrument_dataset_2/ground_truth",

    "MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth",
    "MICCAI/instrument_5_8_training/instrument_dataset_5/ground_truth",

    "MICCAI/instrument_5_8_training/instrument_dataset_6/ground_truth",

    "MICCAI/instrument_5_8_training/instrument_dataset_7/ground_truth",

    "MICCAI/instrument_5_8_training/instrument_dataset_8/ground_truth"






    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/testGT"


]
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
datasetVal = InstrumentDataset(image_dirs=image_dirs_val,gt_dirs=mask_dirs_val,transform=validation_transform,class_to_id=GLOBAL_CLASS_MAPPING)
dataloaderVal = DataLoader(datasetVal,batch_size=batch_size,shuffle=True)

#dataset_cholec = CholecDataset(filtered_ds, transform=train_transform)
datasetMiccai = InstrumentDataset(image_dirs=image_dirs_train,gt_dirs=mask_dirs_train,transform=train_transform,class_to_id=GLOBAL_CLASS_MAPPING,increase = False)


dataloader = DataLoader(datasetMiccai,batch_size=batch_size,shuffle=True,pin_memory=True)
for images, masks in dataloader:
    print(f"Batch di immagini: {images.shape}")  # (batch_size, 3, 224, 224)
    print(f"Batch di maschere: {masks.shape}")  # (batch_size, 1, 224, 224)
    break


#CARICO IL MIO AUTOSAM

device = "cuda" if torch.cuda.is_available() else "cpu"
autosam_checkpoint = "/home/mdezen/distillation/checkpoints/28_07/autoSamFineUnetMUcH0.pth"  # Path to the autosam checkpoint



"""
model = sam_model_registry["autoSamUnet"](checkpoint=None,num_classes=datasetMiccai.getNumClasses())
checkpoint = torch.load(autosam_checkpoint, map_location=device)
state_dict = checkpoint
model_dict = model.state_dict() # Load the state dict into the model

filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(filtered_dict)
model.load_state_dict(model_dict)
"""
checkpoint = torch.load(autosam_checkpoint, map_location=device)
model = sam_model_registry["autoSamUnet"](checkpoint=None, num_classes=1)
model.load_state_dict(checkpoint, strict=False)

"""
encoder_dict = {k.replace("image_encoder.", ""): v
                for k, v in checkpoint.items() if k.startswith("image_encoder.")}

model.image_encoder.load_state_dict(encoder_dict)

model.to(device)
"""
model.to(device=device)
"""
model.train()
for param in model.parameters():
    param.requires_grad = False
for param in model.mask_decoder.parameters():
    param.requires_grad = True
"""
#












lr = 0.0001


optimizer_cfg = {
    'opt': 'adamw',
    'lr': lr,
    'weight_decay': 1e-4,
}
#classifier = InstrumentClassifier(in_channels=3, n_classes=datasetMiccai.getNumClasses()).to(device)
classifier = InstrumentClassifier(in_channels=3, n_classes=datasetMiccai.getNumClasses()).to(device)
classifier.train()
optimizer = create_optimizer_v2(classifier,**optimizer_cfg)
loss_scaler = NativeScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min',factor = 0.1,patience = 3,threshold=0.000001)

epochs = 50
for i, param in enumerate(classifier.parameters()):
    if param.grad is None:
        print(f"Parametro {i} non ha gradiente!")




run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).

    # Set the wandb project where this run will be logged.
    project="autoSamUnetMulti",
    name=name,
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "classifier",
        "dataset": "Miccai ",
        "epochs": epochs,
        "criterion": "CrossEntropyWeighted",
        "batch_size": batch_size,
        "optimizer": optimizer_cfg['opt'],
        "weight_decay": optimizer_cfg['weight_decay'],
        "augmentation": str(train_transform),


    }

)

"""
counts = Counter()
for i in range(len(datasetMiccai)):
    _, mask = datasetMiccai[i]   # mask è un torch.Tensor (H,W)
    unique, values = torch.unique(mask, return_counts=True)
    for u, v in zip(unique.tolist(), values.tolist()):
        if u != 0:
            counts[u] += 1


print("Pixel per classe:", counts)
ordered_counts = [counts.get(cls_id, 0) for cls_id in range(1, len(GLOBAL_CLASS_MAPPING)+1)]
print("Ordered counts:", ordered_counts)

freqs = torch.tensor(ordered_counts, dtype=torch.float32) / sum(ordered_counts)
median_freq = freqs[freqs > 0].median()
weights = median_freq / freqs
print("weights:", weights)

print(weights)
weights = torch.tensor(weights, dtype=torch.float32).to(device)
weights = torch.cat((torch.tensor([0.0], dtype=torch.float32).to(device), weights))  # peso 0 per lo sfondo
"""
criterion = F.cross_entropy#-1 quando ci sono maschere vuote

#TRAINING
patience = 20  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0
checkpoint_path = "checkpoints/01_09/" + name+".pth"

torch.cuda.empty_cache()
gc.collect()

# directory per salvare i risultati
os.makedirs("debug_samples", exist_ok=True)
for epoch in range(0, epochs):
    print(f"Epoch {epoch + 1}/{epochs}")


    train_stats = train_one_epoch_instruments(model,classifier,dataloader,optimizer,device,run,epoch,criterion,loss_scaler)

    torch.cuda.empty_cache()
    gc.collect()
    #print(epoch)
    val_loss = validate_one_epoch_instruments(model,classifier,dataloaderVal,device,run,epoch,criterion)
    scheduler.step(val_loss)  # Update the learning rate scheduler based on validation loss
    print(
        f"Epoch {epoch} loss: {val_loss}")
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(classifier.state_dict(), checkpoint_path)  # Save the best model

    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping condition
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break



