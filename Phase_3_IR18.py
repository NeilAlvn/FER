import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict


# ============================================
# IR18 (ResNet18) Backbone Architecture
# ============================================

class BasicBlock(nn.Module):
    """IR Basic block for ResNet18"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.bn2(out)
        out = self.prelu(out)

        return out


class IR18(nn.Module):
    """ResNet18-IR backbone for face recognition"""
    def __init__(self, num_layers=18, feature_dim=512):
        super(IR18, self).__init__()
        assert num_layers == 18, "Only ResNet18-IR is supported"
        
        self.in_channels = 64
        
        # Input layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        
        # ResNet layers (2, 2, 2, 2 blocks for ResNet18)
        self.layer1 = self._make_layer(64, 2, stride=2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Output layer
        self.bn2 = nn.BatchNorm2d(512 * BasicBlock.expansion)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512 * BasicBlock.expansion * 7 * 7, feature_dim)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn3(x)

        return x


# ============================================
# FAN Model (Frozen for Phase 3)
# ============================================

class FAN(nn.Module):
    """Face Alignment Network - frozen during FER training"""
    def __init__(self, num_landmarks=68):
        super(FAN, self).__init__()
        self.num_landmarks = num_landmarks
        
        from torchvision import models
        mobilenet = models.mobilenet_v2(weights=None)
        self.features = mobilenet.features
        
        self.conv1 = nn.Conv2d(1280, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.heatmap = nn.Conv2d(256, num_landmarks, kernel_size=1)
        
    def forward(self, x):
        features = self.features(x)
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(x))
        heatmaps = self.heatmap(x)
        
        batch_size = heatmaps.size(0)
        heatmaps_flat = heatmaps.reshape(batch_size, self.num_landmarks, -1)
        softmax_maps = F.softmax(heatmaps_flat, dim=2)
        
        h, w = heatmaps.size(2), heatmaps.size(3)
        grid_x = torch.linspace(0, w-1, w).to(x.device)
        grid_y = torch.linspace(0, h-1, h).to(x.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0).reshape(2, -1)
        
        landmarks = torch.matmul(softmax_maps, grid.T.unsqueeze(0))
        
        return landmarks


# ============================================
# Complete FER Model
# ============================================

class FERModel(nn.Module):
    """Complete Facial Expression Recognition Model"""
    def __init__(self, fan_model_path, num_classes=7, feature_dim=512, freeze_fan=True):
        super(FERModel, self).__init__()
        
        # Face Alignment Network (frozen)
        self.fan = FAN(num_landmarks=68)
        checkpoint = torch.load(fan_model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.fan.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.fan.load_state_dict(checkpoint)
        
        if freeze_fan:
            for param in self.fan.parameters():
                param.requires_grad = False
            self.fan.eval()
        
        # IR18 Backbone (changed from IR50)
        self.backbone = IR18(num_layers=18, feature_dim=feature_dim)
        
        # Landmark feature branch
        self.landmark_fc = nn.Sequential(
            nn.Linear(68 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract landmarks (frozen)
        with torch.no_grad():
            landmarks = self.fan(x)
        
        # Extract image features
        img_features = self.backbone(x)
        
        # Process landmarks
        landmarks_flat = landmarks.view(landmarks.size(0), -1)
        landmark_features = self.landmark_fc(landmarks_flat)
        
        # Fuse features
        combined = torch.cat([img_features, landmark_features], dim=1)
        output = self.fusion(combined)
        
        return output


# ============================================
# RAF-DB Dataset Loader
# ============================================

class RAFDBDataset(Dataset):
    """RAF-DB Dataset with pre-extracted landmarks"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load data
        self.landmarks = np.load(os.path.join(root_dir, f'{split}_landmarks.npy'))
        self.labels = np.load(os.path.join(root_dir, f'{split}_labels.npy'))
        
        with open(os.path.join(root_dir, f'{split}_img_names.txt'), 'r') as f:
            self.img_names = f.read().splitlines()
        
        print(f"✓ Loaded {len(self.labels)} samples from {split} set")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.split, self.img_names[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        landmarks = self.landmarks[idx]
        
        return image, label, torch.FloatTensor(landmarks)


# ============================================
# SAM Optimizer
# ============================================

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer"""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        
        self.base_optimizer.step()
        
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like regular optimizers")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# ============================================
# Training Functions
# ============================================

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    model.fan.eval()  # Keep FAN frozen
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for images, labels, landmarks in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # First forward-backward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # Second forward-backward pass
        outputs = model(images)
        criterion(outputs, labels).backward()
        optimizer.second_step(zero_grad=True)
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, landmarks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(val_loader)
    
    return avg_loss, accuracy, all_preds, all_labels


# ============================================
# Main Training Loop
# ============================================

def train_fer_model():
    # Configuration
    PROCESSED_DATA = 'processed_raf_db'
    FAN_MODEL_PATH = 'checkpoints_landmarks/best_fan_model.pth'
    OUTPUT_DIR = 'checkpoints_fer'
    
    BATCH_SIZE = 64
    EPOCHS = 200
    INITIAL_LR = 5e-6
    WEIGHT_DECAY = 1e-4
    LR_DECAY = 0.98
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("PHASE 3: FER MODEL TRAINING (IR18)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Backbone: IR18 (ResNet18-IR)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Initial LR: {INITIAL_LR}")
    print(f"Epochs: {EPOCHS}")
    print("="*60)
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = RAFDBDataset(PROCESSED_DATA, split='train', transform=train_transform)
    val_dataset = RAFDBDataset(PROCESSED_DATA, split='test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    print("\nInitializing model...")
    model = FERModel(FAN_MODEL_PATH, num_classes=7, freeze_fan=True)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer, lr=INITIAL_LR, 
                   weight_decay=WEIGHT_DECAY, rho=0.05)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer.base_optimizer, gamma=LR_DECAY)
    
    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nStarting training...\n")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, 
                                           criterion, DEVICE, epoch)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.base_optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(OUTPUT_DIR, 'best_fer_model.pth'))
            print(f"  ✓ New best model saved! (Acc: {best_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.base_optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth'))
    
    # Plot training curves
    plot_training_curves(history, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {OUTPUT_DIR}/best_fer_model.pth")


def plot_training_curves(history, output_dir):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    print(f"✓ Training curves saved")
    plt.close()


# ============================================
# Run Training
# ============================================

if __name__ == '__main__':
    train_fer_model()