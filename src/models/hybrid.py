# File: src/models/hybrid.py

import torch
import torch.nn as nn
import torchvision.models as models

class HybridDetector(nn.Module):
    def __init__(self, num_classes=2, dct_dim=1024, pretrained=True, freeze_cnn=False, 
                 model_name='mobilenet_v3_small'):
        """
        Lightweight hybrid detector for AI-generated image detection using torchvision models.
        
        Args:
            num_classes: Number of output classes (default: 2)
            dct_dim: Dimension of DCT features (default: 1024)
            pretrained: Use pretrained weights (default: True)
            freeze_cnn: Freeze CNN backbone initially (default: False)
            model_name: CNN backbone architecture (default: 'mobilenet_v3_small')
                Options from torchvision:
                - 'mobilenet_v3_small': ~2.5M params, 576 dim (fastest)
                - 'mobilenet_v3_large': ~5.4M params, 960 dim (balanced)
                - 'mobilenet_v2': ~3.5M params, 1280 dim (classic)
                - 'resnet18': ~11M params, 512 dim (standard)
                - 'resnet34': ~21M params, 512 dim (deeper)
                - 'resnet50': ~25M params, 2048 dim (heavy)
                - 'efficientnet_b0': ~5.3M params, 1280 dim (efficient)
                - 'efficientnet_b1': ~7.8M params, 1280 dim (larger)
        """
        super().__init__()
        
        # Map model names to torchvision functions
        model_dict = {
            'mobilenet_v3_small': models.mobilenet_v3_small,
            'mobilenet_v3_large': models.mobilenet_v3_large,
            'mobilenet_v2': models.mobilenet_v2,
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
        }
        
        # CNN Backbone from torchvision
        try:
            if model_name not in model_dict:
                print(f"⚠️  Unknown model: {model_name}, using mobilenet_v3_small")
                model_name = 'mobilenet_v3_small'
            
            # Load pretrained model
            if pretrained:
                backbone = model_dict[model_name](weights='DEFAULT')
            else:
                backbone = model_dict[model_name](weights=None)
            
            # Extract feature extractor (remove classifier)
            if 'mobilenet' in model_name:
                self.cnn = backbone.features
                # Get output channels from last conv layer
                cnn_out_dim = backbone.classifier[0].in_features
            elif 'resnet' in model_name:
                # Remove avgpool and fc layers
                self.cnn = nn.Sequential(*list(backbone.children())[:-2])
                cnn_out_dim = backbone.fc.in_features
            elif 'efficientnet' in model_name:
                self.cnn = backbone.features
                cnn_out_dim = backbone.classifier[1].in_features
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Add adaptive pooling and flatten
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            
            print(f"   Loaded {model_name} from torchvision")
            print(f"   CNN output dim: {cnn_out_dim}")
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            print("   Falling back to mobilenet_v3_small...")
            backbone = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
            self.cnn = backbone.features
            cnn_out_dim = backbone.classifier[0].in_features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            model_name = 'mobilenet_v3_small'
        
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # MLP for DCT features - simplified
        mlp_out_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(dct_dim, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, mlp_out_dim), 
            nn.BatchNorm1d(mlp_out_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fusion classifier - lighter architecture with dynamic dimension
        fusion_dim = cnn_out_dim + mlp_out_dim
        print(f"   Fusion dim: {cnn_out_dim} (CNN) + {mlp_out_dim} (MLP) = {fusion_dim}")
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Store config for logging
        self.model_name = model_name
        self.cnn_out_dim = cnn_out_dim
        self.mlp_out_dim = mlp_out_dim
        self.fusion_dim = fusion_dim
    
    def forward(self, img_masked, dct_feat):
        # Extract CNN features
        cnn_out = self.cnn(img_masked)
        
        # Pool and flatten (always needed for feature maps)
        cnn_out = self.pool(cnn_out)
        cnn_out = self.flatten(cnn_out)
        
        # Extract MLP features from DCT
        mlp_out = self.mlp(dct_feat)
        
        # Fuse and classify
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        return self.classifier(combined)
    
    def get_model_info(self):
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': self.model_name,
            'cnn_output_dim': self.cnn_out_dim,
            'mlp_output_dim': self.mlp_out_dim,
            'fusion_dim': self.fusion_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params
        }