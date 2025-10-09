"""
Phase 2: Facial Expression Recognition Training with NKF Framework
Target: 93-94% accuracy on RAF-DB
FIXED: All configs now match the research paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import random
from collections import Counter
import cv2


# ============================================================================
# SAM Optimizer (Sharpness-Aware Minimization)
# ============================================================================
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization optimizer
    Paper: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ============================================================================
# Import components from Phase 1
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.conv = nn.Conv2d(in_c, groups, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(groups)
        self.prelu = nn.PReLU(groups)
        self.conv_dw = nn.Conv2d(groups, groups, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn_dw = nn.BatchNorm2d(groups)
        self.prelu_dw = nn.PReLU(groups)
        self.project = nn.Conv2d(groups, out_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn_project = nn.BatchNorm2d(out_c)
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.prelu(self.bn(self.conv(x)))
        x = self.prelu_dw(self.bn_dw(self.conv_dw(x)))
        x = self.bn_project(self.project(x))
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class MobileFaceNet(nn.Module):
    def __init__(self):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = DepthWise(64, 64, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        return out


class FAN(nn.Module):
    """Face Alignment Network"""
    def __init__(self, num_landmarks=68):
        super(FAN, self).__init__()
        self.num_landmarks = num_landmarks
        self.backbone = MobileFaceNet()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.heatmap = nn.Conv2d(128, num_landmarks, kernel_size=1)
        
    def forward(self, x):
        features = self.backbone(x)
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        heatmaps = self.heatmap(x)
        
        # Soft-argmax for landmark coordinates
        batch_size = heatmaps.size(0)
        h, w = heatmaps.size(2), heatmaps.size(3)
        heatmaps_flat = heatmaps.reshape(batch_size, self.num_landmarks, -1)
        softmax_maps = F.softmax(heatmaps_flat, dim=2)
        
        grid_x = torch.linspace(0, w-1, w, device=x.device)
        grid_y = torch.linspace(0, h-1, h, device=x.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0).reshape(2, -1)
        
        landmarks = torch.matmul(softmax_maps, grid.T.unsqueeze(0))
        
        return features, landmarks, heatmaps


# ============================================================================
# ResNet-50 (IR50) Backbone for FER
# ============================================================================
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=2)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


# ============================================================================
# NKF: Keypoint Feature Network
# ============================================================================
class KeypointFeatureNetwork(nn.Module):
    """Extract features at landmark locations"""
    def __init__(self, num_landmarks=68, feature_dim=2048):
        super(KeypointFeatureNetwork, self).__init__()
        self.num_landmarks = num_landmarks
        self.feature_dim = feature_dim
        
    def forward(self, feature_map, landmarks):
        """
        Args:
            feature_map: (B, C, H, W) - features from backbone
            landmarks: (B, 68, 2) - normalized landmark coordinates [0, H-1] or [0, W-1]
        Returns:
            keypoint_features: (B, 68, C)
        """
        B, C, H, W = feature_map.shape
        
        # Normalize landmarks to [-1, 1] for grid_sample
        landmarks_norm = landmarks.clone()
        landmarks_norm[:, :, 0] = 2.0 * landmarks[:, :, 0] / (W - 1) - 1.0
        landmarks_norm[:, :, 1] = 2.0 * landmarks[:, :, 1] / (H - 1) - 1.0
        
        # Reshape for grid_sample: (B, num_landmarks, 1, 2)
        grid = landmarks_norm.unsqueeze(2)  # (B, 68, 1, 2)
        
        # Sample features at landmark locations
        keypoint_features = F.grid_sample(
            feature_map, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )  # (B, C, 68, 1)
        
        keypoint_features = keypoint_features.squeeze(-1).transpose(1, 2)  # (B, 68, C)
        
        return keypoint_features


class RKFA(nn.Module):
    """Representative Keypoint Feature Attention"""
    def __init__(self, feature_dim=2048):
        super(RKFA, self).__init__()
        self.feature_dim = feature_dim
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )
        
    def forward(self, keypoint_features, representative_idx=30):
        """
        Args:
            keypoint_features: (B, 68, C)
            representative_idx: Index of nasal base landmark (default: 30)
        Returns:
            attended_features: (B, 68, C)
        """
        # Get representative feature (nasal base)
        rep_feature = keypoint_features[:, representative_idx:representative_idx+1, :]  # (B, 1, C)
        
        # Compute attention scores
        attention_scores = self.attention(keypoint_features)  # (B, 68, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, 68, 1)
        
        # Apply attention
        attended_features = keypoint_features * attention_weights  # (B, 68, C)
        
        return attended_features


class LandmarkPerturbation(nn.Module):
    """Landmark Perturbation Regularization"""
    def __init__(self, perturbation_scale=0.10):
        super(LandmarkPerturbation, self).__init__()
        self.perturbation_scale = perturbation_scale
        
    def forward(self, landmarks, feature_map_size):
        """
        Add random perturbation to landmarks during training
        """
        if self.training:
            H, W = feature_map_size
            noise = torch.randn_like(landmarks) * self.perturbation_scale
            noise[:, :, 0] *= W
            noise[:, :, 1] *= H
            perturbed_landmarks = landmarks + noise
            
            # Clamp to valid range
            perturbed_landmarks[:, :, 0] = torch.clamp(perturbed_landmarks[:, :, 0], 0, W-1)
            perturbed_landmarks[:, :, 1] = torch.clamp(perturbed_landmarks[:, :, 1], 0, H-1)
            
            return perturbed_landmarks
        else:
            return landmarks


# ============================================================================
# Complete NKF Model
# ============================================================================
class NKF(nn.Module):
    """Complete NKF Framework for FER"""
    def __init__(self, num_classes=7, num_landmarks=68, use_rkfa=True, 
                 perturbation_scale=0.10, fan_checkpoint=None):
        super(NKF, self).__init__()
        
        # Face Alignment Network (frozen during FER training)
        self.fan = FAN(num_landmarks=num_landmarks)
        if fan_checkpoint:
            print(f"Loading FAN from {fan_checkpoint}")
            checkpoint = torch.load(fan_checkpoint, map_location='cpu')
            self.fan.load_state_dict(checkpoint['model_state_dict'])
            print("✅ FAN loaded successfully")
        
        # Freeze FAN
        for param in self.fan.parameters():
            param.requires_grad = False
        
        # Backbone for FER (ResNet-50)
        self.backbone = ResNet50()
        
        # Keypoint Feature Extraction
        self.kf_net = KeypointFeatureNetwork(num_landmarks=num_landmarks, feature_dim=2048)
        
        # RKFA
        self.use_rkfa = use_rkfa
        if use_rkfa:
            self.rkfa = RKFA(feature_dim=2048)
        
        # Landmark Perturbation
        self.perturbation = LandmarkPerturbation(perturbation_scale=perturbation_scale)
        
        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_kf = nn.Linear(num_landmarks * 2048, 2048)
        self.fc_global = nn.Linear(2048, 2048)
        self.fc_final = nn.Linear(2048 * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        B = x.size(0)
        
        # 1. Face Alignment (frozen)
        with torch.no_grad():
            self.fan.eval()
            _, landmarks, _ = self.fan(x)
        
        # 2. Extract global features from backbone
        global_features = self.backbone(x)  # (B, 2048, H', W')
        H, W = global_features.size(2), global_features.size(3)
        
        # 3. Apply landmark perturbation (training only)
        landmarks_perturbed = self.perturbation(landmarks, (H, W))
        
        # 4. Extract keypoint features
        keypoint_features = self.kf_net(global_features, landmarks_perturbed)  # (B, 68, 2048)
        
        # 5. Apply RKFA
        if self.use_rkfa:
            keypoint_features = self.rkfa(keypoint_features)  # (B, 68, 2048)
        
        # 6. Global pooling for global features
        global_pooled = self.global_pool(global_features).view(B, -1)  # (B, 2048)
        global_pooled = self.fc_global(global_pooled)
        
        # 7. Flatten keypoint features
        kf_flattened = keypoint_features.reshape(B, -1)  # (B, 68*2048)
        kf_reduced = self.fc_kf(kf_flattened)  # (B, 2048)
        
        # 8. Concatenate and classify
        combined = torch.cat([global_pooled, kf_reduced], dim=1)  # (B, 4096)
        combined = self.dropout(combined)
        output = self.fc_final(combined)  # (B, num_classes)
        
        return output


# ============================================================================
# RAF-DB Dataset for FER
# ============================================================================
class RAFDBDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=112):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.data = []
        
        # Emotion mapping
        self.emotion_names = {
            1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happiness",
            5: "Sadness", 6: "Anger", 7: "Neutral"
        }
        
        split_dir = os.path.join(root_dir, split)
        
        # Load all images
        for emotion_id in range(1, 8):
            emotion_dir = os.path.join(split_dir, str(emotion_id))
            if not os.path.exists(emotion_dir):
                continue
            
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_name in image_files:
                img_path = os.path.join(emotion_dir, img_name)
                self.data.append({
                    'image_path': img_path,
                    'label': emotion_id - 1  # 0-indexed for PyTorch
                })
        
        print(f"{split} set: {len(self.data)} images")
        
        # Print class distribution
        labels = [d['label'] for d in self.data]
        label_counts = Counter(labels)
        for label in sorted(label_counts.keys()):
            print(f"  Class {label} ({self.emotion_names[label+1]}): {label_counts[label]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = sample['label']
        
        return image, label


# ============================================================================
# Data Augmentation (Following Paper)
# ============================================================================
def get_fer_transforms(img_size=112, is_train=True):
    """Multi-stage data augmentation as per paper"""
    if is_train:
        transform = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.15, 0.15),
                    scale=(0.85, 1.15)
                )
            ], p=0.8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))  # ✅ FIXED: was (0.02, 0.15)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


# ============================================================================
# Training Functions with SAM
# ============================================================================
def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # First forward-backward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # SAM requires closure for second forward-backward pass
        def closure():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        optimizer.zero_grad()
        
        # Statistics
        with torch.no_grad():
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{acc:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{acc:.2f}%'
            })
    
    return running_loss / len(val_loader), 100. * correct / total


# ============================================================================
# Main Training Script
# ============================================================================
def main():
    CONFIG = {
        'data_root': './archive/DATASET',
        'fan_checkpoint': './checkpoints_rafdb_landmarks/best_fan_rafdb.pth',
        'img_size': 112,
        'batch_size': 64,
        'num_epochs': 200,                   
        'learning_rate': 5e-5,             
        'weight_decay': 1e-4,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_fer_rafdb',
        'use_rkfa': True,
        'perturbation_scale': 0.10,           # ✅ FIXED: was 0.05
        'sam_rho': 0.05,                      # ✅ ADDED: SAM parameter
        'use_class_balanced_sampling': False,
    }
    
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    print("="*70)
    print("Phase 2: Facial Expression Recognition Training (NKF)")
    print("="*70)
    print(f"Device: {CONFIG['device']}")
    print(f"Dataset: RAF-DB")
    print(f"FAN Checkpoint: {CONFIG['fan_checkpoint']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Use RKFA: {CONFIG['use_rkfa']}")
    print(f"Perturbation scale: {CONFIG['perturbation_scale']}")
    print(f"SAM rho: {CONFIG['sam_rho']}")
    print("="*70)
    
    # Datasets
    train_transform = get_fer_transforms(CONFIG['img_size'], is_train=True)
    val_transform = get_fer_transforms(CONFIG['img_size'], is_train=False)
    
    train_dataset = RAFDBDataset(CONFIG['data_root'], 'train', train_transform, CONFIG['img_size'])
    val_dataset = RAFDBDataset(CONFIG['data_root'], 'test', val_transform, CONFIG['img_size'])
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    # Model
    model = NKF(
        num_classes=7,
        num_landmarks=68,
        use_rkfa=CONFIG['use_rkfa'],
        perturbation_scale=CONFIG['perturbation_scale'],
        fan_checkpoint=CONFIG['fan_checkpoint']
    )
    model = model.to(CONFIG['device'])
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # SAM Optimizer (wrapping Adam as per paper)
    base_optimizer = lambda params, **kwargs: optim.Adam(
        params, 
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        **kwargs
    )
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        rho=CONFIG['sam_rho'],
        adaptive=False
    )
    
    # Learning rate scheduler (exponential decay 0.98 per epoch)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer.base_optimizer, gamma=0.98)
    
    # Training loop
    best_acc = 0.0
    
    print("\n🚀 Starting training...")
    print(f"Training for {CONFIG['num_epochs']} epochs to reach 93-94% accuracy target!\n")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'], epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.base_optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
            }, os.path.join(CONFIG['save_dir'], 'best_nkf_rafdb.pth'))
            print(f"✅ Best model saved! Accuracy: {val_acc:.2f}%")
        
        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
            }, os.path.join(CONFIG['save_dir'], f'checkpoint_epoch_{epoch}.pth'))
            print(f"💾 Checkpoint saved at epoch {epoch}")
    
    print("\n" + "="*70)
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    print(f"Target: 93-94% ({'✅ REACHED' if best_acc >= 93.0 else '⚠️ Continue training'})")
    print("="*70)


if __name__ == '__main__':
    main()