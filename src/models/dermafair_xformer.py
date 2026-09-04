import torch
import torch.nn as nn
import torchvision.models as models


class DermaFairXFormer(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # CNN branch: ConvNeXt-Tiny
        cnn = models.convnext_tiny(weights=None)

        self.cnn_features = cnn.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # ConvNeXt-Tiny feature dimension = 768
        self.cnn_projection = nn.Linear(768, 256)

        # Transformer branch: Swin-Tiny
        swin = models.swin_t(weights=None)

        self.transformer_features = swin.features
        self.transformer_norm = swin.norm

        # Swin-Tiny feature dimension = 768
        self.transformer_pool = nn.AdaptiveAvgPool2d(1)
        self.transformer_projection = nn.Linear(768, 256)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Lesion-aware attention
        self.lesion_attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, lesion_mask=None):

        # CNN branch
        cnn_features = self.cnn_features(x)
        cnn_features = self.cnn_pool(cnn_features)
        cnn_features = torch.flatten(cnn_features, 1)
        cnn_features = self.cnn_projection(cnn_features)

        # Transformer branch
        transformer_features = self.transformer_features(x)

        # Swin output: [B, H, W, C] -> [B, C, H, W]
        transformer_features = transformer_features.permute(
            0, 3, 1, 2
        )

        transformer_features = self.transformer_norm(
            transformer_features.permute(0, 2, 3, 1)
        )

        transformer_features = transformer_features.permute(
            0, 3, 1, 2
        )

        transformer_features = self.transformer_pool(
            transformer_features
        )

        transformer_features = torch.flatten(
            transformer_features, 1
        )

        transformer_features = self.transformer_projection(
            transformer_features
        )

        # Complementary feature fusion
        fused = torch.cat(
            [cnn_features, transformer_features],
            dim=1
        )

        fused = self.fusion(fused)

        # Lesion-aware feature weighting
        attention = self.lesion_attention(fused)

        if lesion_mask is not None:
            mask_weight = lesion_mask.mean(
                dim=(1, 2, 3),
                keepdim=False
            ).unsqueeze(1)

            attention = attention * mask_weight

        fused = fused * attention

        # Classification
        logits = self.classifier(fused)

        return logits
