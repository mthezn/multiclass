import torch
import torch.nn as nn
import torch.nn.functional as F
"""
class InstrumentClassifier(nn.Module):
    def __init__(self, in_channels=3, n_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),  # global pooling
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x 
    """



#version con pooling per calssifcare usando tutta l'imagine ma concentrandosi sullo strumento
class InstrumentClassifier(nn.Module):
    def __init__(self, in_channels=3, n_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,64 , kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),
        )
        # Dropout per ridurre overfitting
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, mask):

        """x:    (B,3,H,W) immagine
        mask: (B,1,H,W) maschera binaria per l'istanza
        """
        feat = self.features(x)  # (B,512,h,w)
        mask_resized = F.interpolate(mask, size=feat.shape[2:], mode="nearest")  # (B,1,h,w)

        # masked average pooling
        feat_masked = feat * mask_resized
        pooled = feat_masked.sum(dim=(2,3)) / (mask_resized.sum(dim=(2,3)) + 1e-6)  # (B,64)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)  # (B,n_classes)
        return out



"""

class InstrumentClassifier(nn.Module):
    def __init__(self, in_channels=32, num_classes=7, dropout_p=0.3):
        super().__init__()

        # 4 blocchi Conv + Norm + ReLU
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Max pooling spaziale con stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout per ridurre overfitting
        self.dropout = nn.Dropout(dropout_p)

        # Fully connected dopo global pooling
        self.fc = nn.Linear(256, num_classes)

    def forward(self, feats, masks):
        
        feats: (B, C, H, W) feature map dal decoder (es. 32 canali)
        masks: (B, 1, H, W) maschere binarie per istanza
      
        # Mascheratura: element-wise product
        masked_feats = feats * masks

        # Passaggio nella CNN
        x = self.conv_block(masked_feats)

        # Pooling spaziale
        x = self.pool(x)

        # Global max pooling (riduce H,W -> 1,1)
        x = F.adaptive_max_pool2d(x, (1, 1))  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)

        # Dropout
        x = self.dropout(x)

        # Classificazione
        logits = self.fc(x)
        return logits
        """
