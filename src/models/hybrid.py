import torch
import torch.nn as nn
import timm
import warnings

class HybridDetector(nn.Module):
    def __init__(self, num_classes=2, dct_dim=1024, pretrained=True, freeze_cnn=False):
        super().__init__()
        
        try:
            self.cnn = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        except:
            print("Pretrained failed, using random init")
            self.cnn = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(dct_dim, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True)
        )
        
        fusion_dim = 1280 + 256
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, img_masked, dct_feat):
        cnn_out = self.cnn(img_masked)
        mlp_out = self.mlp(dct_feat)
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        return self.classifier(combined)