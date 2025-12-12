import copy
import albumentations as A
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import Weights
from wandb.integration.sklearn.plot.classifier import classifier

from Dataset import CholecDataset
from Dataset import InstrumentDataset,InstrumentDatasetTest
from instrumentClassifier import InstrumentClassifier, SurgicalToolClassifier
from modeling.build_sam import sam_model_registry
from Dataset import ImageMaskDataset
from collections import Counter

from utils import *
from albumentations.pytorch import ToTensorV2
import wandb
import numpy as np
import torch.nn as nn
from engine import train_one_epoch_instruments_efficientnet, validate_efficientnet
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
batch_size = 2
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
image_dirs_val = ["MICCAImod/instrument_1_4_training/validation_dataset_1/left_frames"]
mask_dirs_val = ["MICCAImod/instrument_1_4_training/validation_dataset_1/ground_truth"
                ]

image_dirs_train = [
    #"MICCAImod/instrument_1_4_training/instrument_dataset_1/left_frames",

    #"MICCAImod/instrument_1_4_training/instrument_dataset_2/left_frames",
    #"MICCAImod/instrument_1_4_training/instrument_dataset_3/left_frames",

    #"MICCAImod/instrument_1_4_training/instrument_dataset_4/left_frames",
    #"MICCAImod/instrument_5_8_training/instrument_dataset_5/left_frames",
    #"MICCAImod/instrument_5_8_training/instrument_dataset_6/left_frames",
    #"MICCAImod/instrument_5_8_training/instrument_dataset_7/left_frames",
    "MICCAImod/instrument_5_8_training/instrument_dataset_8/left_frames",
]
mask_dirs_train = [
    #"/home/mdezen/multiclass/MICCAI/instrument_1_4_training/instrument_dataset_1a/ground_truth",
    #"MICCAImod/instrument_1_4_training/instrument_dataset_1/ground_truth",

    #"MICCAImod/instrument_1_4_training/instrument_dataset_2/ground_truth",
    #"MICCAImod/instrument_1_4_training/instrument_dataset_3/ground_truth"
    #"MICCAImod/instrument_1_4_training/instrument_dataset_4/ground_truth",
    #"MICCAImod/instrument_5_8_training/instrument_dataset_5/ground_truth",

    #"MICCAImod/instrument_5_8_training/instrument_dataset_6/ground_truth",

    #"MICCAImod/instrument_5_8_training/instrument_dataset_7/ground_truth",

    "MICCAImod/instrument_5_8_training/instrument_dataset_8/ground_truth"






    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/testGT"


]
GLOBAL_CLASS_MAPPING = {
    'Large_Needle_Driver': 1,
    'Prograsp_Forceps': 2,

    'Bipolar_Forceps': 2,#cambia a 2 se vuoi uificare le forceps
    'Grasping_Retractor': 3,
    'Maryland_Bipolar_Forceps': 4,
    'Monopolar_Curved_Scissors': 5,
    'Other': 6,
    'Vessel_Sealer': 7
}
#datasetVal = InstrumentDataset(image_dirs=image_dirs_val,gt_dirs=mask_dirs_val,transform=validation_transform,class_to_id=GLOBAL_CLASS_MAPPING)
datasetVal = InstrumentDataset(image_dirs=image_dirs_val,gt_dirs=mask_dirs_val,transform=validation_transform,class_to_id=GLOBAL_CLASS_MAPPING)
dataloaderVal = DataLoader(datasetVal,batch_size=batch_size,shuffle=False)
print(len(datasetVal))

#dataset_cholec = CholecDataset(filtered_ds, transform=train_transform)
datasetMiccai = InstrumentDataset(image_dirs=image_dirs_train,gt_dirs=mask_dirs_train,transform=train_transform,class_to_id=GLOBAL_CLASS_MAPPING,increase = False)



#CARICO IL MIO AUTOSAM

device = "cuda" if torch.cuda.is_available() else "cpu"
autosam_checkpoint = "/home/shared-nearmrs/mdezenDatasets/autoSamFineUnetMUcH0.pth"  # Path to the autosam checkpoint



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












lr = 0.001



#classifier = InstrumentClassifier(in_channels=3, n_classes=datasetMiccai.getNumClasses()).to(device)
classifier = SurgicalToolClassifier(num_classes=6,pretrained=True,dropout=0.4).to(device)
classifier.train()
optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=lr,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

loss_scaler = NativeScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

epochs = 100




run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).

    # Set the wandb project where this run will be logged.
    project="autoSamUnetMulti",
    name=name,
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "classifierEfficientNet",
        "dataset": "Miccai ",
        "epochs": epochs,
        "criterion": "CrossENTROPY",
        "batch_size": batch_size,
        "optimizer": "adamW",
        "weight_decay": "1e-4",
        "augmentation": str(train_transform),


    }

)


counts = Counter()
for i in range(len(datasetVal)):
    _, mask = datasetVal[i]   # mask è un torch.Tensor (H,W)
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

criterion = torch.nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)#-1 quando ci sono maschere vuote

#TRAINING
patience = 20  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0
checkpoint_path = "checkpoints/01_09/" + name+".pth"

torch.cuda.empty_cache()
gc.collect()
#sampler = WeightedRandomSampler(weights=weights, num_samples=len(datasetMiccai), replacement=True)
dataloader = DataLoader(datasetMiccai,batch_size=batch_size,pin_memory=True,num_workers=4,shuffle=True)
# directory per salvare i risultati
os.makedirs("debug_samples", exist_ok=True)
for epoch in range(0, epochs):
    print(f"Epoch {epoch + 1}/{epochs}")


    train_stats = train_one_epoch_instruments_efficientnet(model,classifier,dataloader,optimizer,device,run,epoch,criterion,counts,loss_scaler=None,use_mixup=True,focal_loss=False,roi_size=256)

    torch.cuda.empty_cache()
    gc.collect()
    scheduler.step()
    #print(epoch)
    if epoch % 5 == 0:

        val_loss = validate_efficientnet(model,classifier,dataloaderVal,device,criterion,roi_size= 256,run=run,epoch=epoch)
          # Update the learning rate scheduler based on validation loss
        print(
            f"Epoch {epoch} loss: {val_loss}")
        if val_loss["loss"] < best_val_loss:
            loss = val_loss["loss"]
            print(f"Validation loss improved from {best_val_loss:.4f} to {loss:.4f}. Saving model...")
            best_val_loss = loss
            epochs_no_improve = 0
            torch.save(classifier.state_dict(), checkpoint_path)  # Save the best model

        else:
            epochs_no_improve += 5
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping condition
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break



