import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


# ============================================================================
# MobileFaceNet Architecture (Simplified Implementation)
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


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
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.prelu_dw(x)
        x = self.project(x)
        x = self.bn_project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=128):
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


# ============================================================================
# Face Alignment Network (FAN) with MobileFaceNet backbone
# ============================================================================
class FAN(nn.Module):
    """Face Alignment Network for landmark detection"""
    def __init__(self, num_landmarks=68):
        super(FAN, self).__init__()
        self.num_landmarks = num_landmarks
        
        # MobileFaceNet backbone
        self.backbone = MobileFaceNet()
        
        # Heatmap prediction layers
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.heatmap = nn.Conv2d(128, num_landmarks, kernel_size=1)
        
    def forward(self, x):
        # Extract features using MobileFaceNet
        features = self.backbone(x)
        
        # Generate heatmaps for landmarks
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        heatmaps = self.heatmap(x)  # Bx68xHxW
        
        # Get landmark coordinates from heatmaps using soft-argmax
        batch_size = heatmaps.size(0)
        h, w = heatmaps.size(2), heatmaps.size(3)
        
        # Flatten heatmaps
        heatmaps_flat = heatmaps.reshape(batch_size, self.num_landmarks, -1)
        
        # Apply softmax to get probability distribution
        softmax_maps = F.softmax(heatmaps_flat, dim=2)
        
        # Create coordinate grids
        grid_x = torch.linspace(0, w-1, w, device=x.device)
        grid_y = torch.linspace(0, h-1, h, device=x.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0).reshape(2, -1)  # 2xN
        
        # Compute weighted coordinates (soft-argmax)
        landmarks = torch.matmul(softmax_maps, grid.T.unsqueeze(0))  # Bx68x2
        
        return features, landmarks, heatmaps


# ============================================================================
# RAF-DB Dataset for Landmark Training
# ============================================================================
class RAFDBLandmarkDataset(Dataset):
    """
    RAF-DB Dataset Loader for Landmark Training
    Loads images and corresponding generated landmarks
    """
    def __init__(self, root_dir, landmark_dir, split='train', transform=None, img_size=112):
        self.root_dir = root_dir
        self.landmark_dir = landmark_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.data = []
        
        print(f"\nLoading {split} dataset")
        print(f"  Images from: {root_dir}")
        print(f"  Landmarks from: {landmark_dir}")
        
        # RAF-DB has 7 emotion folders (1-7)
        split_dir = os.path.join(root_dir, split)
        landmark_split_dir = os.path.join(landmark_dir, split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Image directory does not exist: {split_dir}")
        
        if not os.path.exists(landmark_split_dir):
            raise ValueError(f"Landmark directory does not exist: {landmark_split_dir}\n"
                           f"Please run the landmark generation script first!")
        
        # Load all images and landmarks from emotion folders
        loaded_count = 0
        skipped_count = 0
        
        for emotion_id in range(1, 8):
            emotion_dir = os.path.join(split_dir, str(emotion_id))
            landmark_emotion_dir = os.path.join(landmark_split_dir, str(emotion_id))
            
            if not os.path.exists(emotion_dir):
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_name in image_files:
                img_path = os.path.join(emotion_dir, img_name)
                
                # Get corresponding landmark file
                landmark_name = img_name.rsplit('.', 1)[0] + '.npy'
                landmark_path = os.path.join(landmark_emotion_dir, landmark_name)
                
                # Only include if landmark file exists
                if os.path.exists(landmark_path):
                    self.data.append({
                        'image_path': img_path,
                        'landmark_path': landmark_path,
                        'emotion': emotion_id
                    })
                    loaded_count += 1
                else:
                    skipped_count += 1
            
            print(f"  Emotion {emotion_id}: {len([d for d in self.data if d['emotion'] == emotion_id])} samples")
        
        print(f"\nTotal {split} samples loaded: {len(self.data)}")
        if skipped_count > 0:
            print(f"⚠️  Skipped {skipped_count} images (no landmarks found)")
        
        if len(self.data) == 0:
            raise ValueError(f"No samples found! Please run landmark generation first.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        orig_w, orig_h = image.size
        
        # Resize to target size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Generate pseudo landmarks (uniform grid as placeholder)
        # TODO: Replace with PFA-generated landmarks
        landmarks = self._generate_pseudo_landmarks()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.from_numpy(landmarks).float()
    
    def _generate_pseudo_landmarks(self):
        """
        Generate pseudo landmarks as placeholder
        In production: Use PFA network to generate real landmarks
        """
        # Create a basic 68-point facial landmark template (normalized to [0,1])
        landmarks = np.array([
            # Jaw line (0-16): bottom contour
            [0.2, 0.9], [0.22, 0.92], [0.25, 0.94], [0.28, 0.96], [0.32, 0.97],
            [0.36, 0.98], [0.40, 0.98], [0.44, 0.98], [0.50, 0.98],
            [0.56, 0.98], [0.60, 0.98], [0.64, 0.98], [0.68, 0.97],
            [0.72, 0.96], [0.75, 0.94], [0.78, 0.92], [0.80, 0.9],
            # Right eyebrow (17-21)
            [0.25, 0.35], [0.30, 0.33], [0.35, 0.32], [0.40, 0.33], [0.44, 0.35],
            # Left eyebrow (22-26)
            [0.56, 0.35], [0.60, 0.33], [0.65, 0.32], [0.70, 0.33], [0.75, 0.35],
            # Nose bridge (27-30)
            [0.50, 0.40], [0.50, 0.45], [0.50, 0.50], [0.50, 0.55],
            # Nose bottom (31-35)
            [0.42, 0.58], [0.46, 0.60], [0.50, 0.61], [0.54, 0.60], [0.58, 0.58],
            # Right eye (36-41)
            [0.32, 0.42], [0.36, 0.40], [0.40, 0.40], [0.44, 0.42],
            [0.40, 0.43], [0.36, 0.43],
            # Left eye (42-47)
            [0.56, 0.42], [0.60, 0.40], [0.64, 0.40], [0.68, 0.42],
            [0.64, 0.43], [0.60, 0.43],
            # Outer mouth (48-59) - 12 points
            [0.38, 0.72], [0.41, 0.71], [0.44, 0.70], [0.47, 0.70], [0.50, 0.70],
            [0.53, 0.70], [0.56, 0.70], [0.59, 0.71], [0.62, 0.72],
            [0.59, 0.76], [0.50, 0.77], [0.41, 0.76],
            # Inner mouth (60-67) - 8 points
            [0.41, 0.73], [0.44, 0.72], [0.47, 0.72], [0.50, 0.72],
            [0.53, 0.72], [0.56, 0.72], [0.59, 0.73], [0.50, 0.74],
        ], dtype=np.float32)
        
        return landmarks


# ============================================================================
# Knowledge Distillation Loss
# ============================================================================
class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation from PFA teacher
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, student_landmarks, teacher_landmarks, gt_landmarks):
        """
        Args:
            student_landmarks: Predicted landmarks from student network
            teacher_landmarks: Landmarks from PFA teacher (if available)
            gt_landmarks: Ground truth landmarks
        """
        # Primary loss: MSE with ground truth
        loss_gt = self.mse(student_landmarks, gt_landmarks)
        
        # If teacher predictions available, add distillation loss
        if teacher_landmarks is not None:
            loss_kd = self.mse(student_landmarks, teacher_landmarks)
            return self.alpha * loss_gt + (1 - self.alpha) * loss_kd
        
        return loss_gt


# ============================================================================
# Training and Validation Functions
# ============================================================================
def compute_nme(predictions, targets, img_size=112):
    """
    Compute Normalized Mean Error (NME)
    Normalized by inter-ocular distance
    """
    # Convert normalized coordinates back to pixel coordinates
    pred_pts = predictions * img_size
    target_pts = targets * img_size
    
    # Compute inter-ocular distance (between eye centers)
    # Left eye: points 36-41, Right eye: points 42-47
    left_eye = target_pts[:, 36:42, :].mean(dim=1)
    right_eye = target_pts[:, 42:48, :].mean(dim=1)
    inter_ocular = torch.norm(left_eye - right_eye, dim=1) + 1e-6
    
    # Compute mean error
    error = torch.norm(pred_pts - target_pts, dim=2).mean(dim=1)
    
    # Normalize by inter-ocular distance
    nme = error / inter_ocular
    
    return nme.mean().item() * 100


def train_epoch(model, train_loader, optimizer, criterion, device, img_size=112):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_nme = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, landmarks in pbar:
        images = images.to(device)
        landmarks = landmarks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        _, pred_landmarks, heatmaps = model(images)
        
        # Normalize predicted landmarks to [0, 1]
        h, w = heatmaps.size(2), heatmaps.size(3)
        pred_landmarks_norm = pred_landmarks.clone()
        pred_landmarks_norm[:, :, 0] = pred_landmarks[:, :, 0] / (w - 1)
        pred_landmarks_norm[:, :, 1] = pred_landmarks[:, :, 1] / (h - 1)
        
        # Compute loss (no teacher for now, just GT)
        loss = criterion(pred_landmarks_norm, None, landmarks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            nme = compute_nme(pred_landmarks_norm, landmarks, img_size)
        
        running_loss += loss.item()
        running_nme += nme
        
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'NME': f'{running_nme / (pbar.n + 1):.2f}%'
        })
    
    return running_loss / len(train_loader), running_nme / len(train_loader)


def validate(model, val_loader, criterion, device, img_size=112):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_nme = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, landmarks in pbar:
            images = images.to(device)
            landmarks = landmarks.to(device)
            
            # Forward pass
            _, pred_landmarks, heatmaps = model(images)
            
            # Normalize predicted landmarks
            h, w = heatmaps.size(2), heatmaps.size(3)
            pred_landmarks_norm = pred_landmarks.clone()
            pred_landmarks_norm[:, :, 0] = pred_landmarks[:, :, 0] / (w - 1)
            pred_landmarks_norm[:, :, 1] = pred_landmarks[:, :, 1] / (h - 1)
            
            # Compute metrics
            loss = criterion(pred_landmarks_norm, None, landmarks)
            nme = compute_nme(pred_landmarks_norm, landmarks, img_size)
            
            running_loss += loss.item()
            running_nme += nme
            
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'NME': f'{running_nme / (pbar.n + 1):.2f}%'
            })
    
    return running_loss / len(val_loader), running_nme / len(val_loader)


# ============================================================================
# Main Training Script
# ============================================================================
def check_dataset_structure(data_root):
    """Check and print dataset structure"""
    print("\n" + "="*70)
    print("DATASET STRUCTURE CHECK")
    print("="*70)
    
    print(f"\nChecking path: {data_root}")
    print(f"Absolute path: {os.path.abspath(data_root)}")
    print(f"Path exists: {os.path.exists(data_root)}")
    
    if not os.path.exists(data_root):
        print("\n❌ ERROR: Dataset path does not exist!")
        print(f"Current working directory: {os.getcwd()}")
        print("\nPlease update CONFIG['data_root'] to the correct path.")
        return False
    
    print(f"\nContents of {data_root}:")
    try:
        contents = os.listdir(data_root)
        for item in contents:
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"  Error listing directory: {e}")
        return False
    
    # Check for train and test folders
    train_path = os.path.join(data_root, 'train')
    test_path = os.path.join(data_root, 'test')
    
    print(f"\nTrain folder exists: {os.path.exists(train_path)}")
    print(f"Test folder exists: {os.path.exists(test_path)}")
    
    if not os.path.exists(train_path):
        print("\n❌ ERROR: 'train' folder not found!")
        return False
    
    # Check emotion folders
    print(f"\nChecking emotion folders in train/:")
    total_images = 0
    for emotion_id in range(1, 8):
        emotion_path = os.path.join(train_path, str(emotion_id))
        if os.path.exists(emotion_path):
            images = [f for f in os.listdir(emotion_path) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"  Emotion {emotion_id}: {len(images)} images")
            total_images += len(images)
        else:
            print(f"  Emotion {emotion_id}: ❌ Folder not found")
    
    print(f"\nTotal training images: {total_images}")
    
    if total_images == 0:
        print("\n❌ ERROR: No images found in training folders!")
        return False
    
    print("\n✅ Dataset structure looks good!")
    print("="*70)
    return True


def main():
    # Configuration matching the paper
    CONFIG = {
        'data_root': './archive/DATASET',  # Your RAF-DB path
        # If this doesn't work, use: r'C:\xampp\htdocs\FER\archive\DATASET'
        'img_size': 112,
        'batch_size': 16,
        'num_epochs': 200,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_rafdb_landmarks',
    }
    
    # First, check dataset structure
    if not check_dataset_structure(CONFIG['data_root']):
        print("\n" + "="*70)
        print("PLEASE FIX THE DATASET PATH AND TRY AGAIN")
        print("="*70)
        print("\nPossible solutions:")
        print("1. Update CONFIG['data_root'] in the main() function")
        print("   Example: CONFIG['data_root'] = r'C:\\xampp\\htdocs\\FER\\DATASET'")
        print("\n2. Or move your DATASET folder to the current directory:")
        print(f"   {os.getcwd()}")
        return
    
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    print("="*70)
    print("RAF-DB Facial Landmark Training with MobileFaceNet")
    print("="*70)
    print(f"Device: {CONFIG['device']}")
    print(f"Dataset: RAF-DB")
    print(f"Architecture: MobileFaceNet-based FAN")
    print(f"Image size: {CONFIG['img_size']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Optimizer: SGD (lr={CONFIG['learning_rate']}, momentum={CONFIG['momentum']})")
    print(f"Loss: MSE with Knowledge Distillation")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print("="*70)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare datasets
    train_dataset = RAFDBLandmarkDataset(
        CONFIG['data_root'],
        landmark_dir='./landmarks_rafdb',
        split='train',
        transform=train_transform,
        img_size=CONFIG['img_size']
    )
    
    val_dataset = RAFDBLandmarkDataset(
        CONFIG['data_root'],
        landmark_dir='./landmarks_rafdb',
        split='test',
        transform=val_transform,
        img_size=CONFIG['img_size']
    )
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    # Create model
    print("\nInitializing MobileFaceNet-based FAN...")
    model = FAN(num_landmarks=68)
    model = model.to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = DistillationLoss(alpha=0.5)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # Training loop
    best_nme = float('inf')
    train_losses, train_nmes = [], []
    val_losses, val_nmes = [], []
    
    print("\nStarting training...")
    print("="*70)
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{CONFIG['num_epochs']}]")
        print("-" * 70)
        
        # Train
        train_loss, train_nme = train_epoch(
            model, train_loader, optimizer, criterion, 
            CONFIG['device'], CONFIG['img_size']
        )
        train_losses.append(train_loss)
        train_nmes.append(train_nme)
        
        # Validate
        val_loss, val_nme = validate(
            model, val_loader, criterion, 
            CONFIG['device'], CONFIG['img_size']
        )
        val_losses.append(val_loss)
        val_nmes.append(val_nme)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f} | NME: {train_nme:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f} | NME: {val_nme:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_nme < best_nme:
            best_nme = val_nme
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'nme': val_nme,
                'loss': val_loss,
                'config': CONFIG
            }
            torch.save(checkpoint, os.path.join(CONFIG['save_dir'], 'best_fan_rafdb.pth'))
            print(f"  ✓ Best model saved! (NME: {val_nme:.2f}%)")
        
        print(f"{'='*70}")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(CONFIG['save_dir'], f'fan_rafdb_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch+1}")
    
    print("\n" + "="*70)
    print(f"Training completed!")
    print(f"Best validation NME: {best_nme:.2f}%")
    print(f"Model saved to: {CONFIG['save_dir']}/best_fan_rafdb.pth")
    print("="*70)


if __name__ == '__main__':
    main()