import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


class FAN(nn.Module):
    """Face Alignment Network for landmark detection"""
    def __init__(self, num_landmarks=68):
        super(FAN, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Using MobileNetV2 as backbone for FAN
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        
        # Heatmap prediction layers
        self.conv1 = nn.Conv2d(1280, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.heatmap = nn.Conv2d(256, num_landmarks, kernel_size=1)
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Generate heatmaps for landmarks
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(x))
        heatmaps = self.heatmap(x)  # Bx68xHxW
        
        # Get landmark coordinates from heatmaps
        batch_size = heatmaps.size(0)
        heatmaps_flat = heatmaps.reshape(batch_size, self.num_landmarks, -1)
        
        # Soft-argmax to get coordinates
        softmax_maps = F.softmax(heatmaps_flat, dim=2)
        
        # Create coordinate grids
        h, w = heatmaps.size(2), heatmaps.size(3)
        grid_x = torch.linspace(0, w-1, w).to(x.device)
        grid_y = torch.linspace(0, h-1, h).to(x.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0).reshape(2, -1)  # 2xN
        
        # Compute weighted coordinates
        landmarks = torch.matmul(softmax_maps, grid.T.unsqueeze(0))  # Bx68x2
        
        return features, landmarks, heatmaps


class W300Dataset(Dataset):
    """
    300W Dataset Loader
    Supports both train and test splits
    """
    def __init__(self, root_dir, split='train', transform=None, img_size=112):
        """
        Args:
            root_dir: Path to 300W dataset root
            split: 'train' or 'test'
            transform: Optional transform
            img_size: Target image size
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.data = []
        
        # Define folders based on split
        if split == 'train':
            # 300W training set
            folders = ['afw', 'helen/trainset', 'lfpw/trainset']
            xml_file = 'labels_ibug_300W_train.xml'
        else:
            # 300W test sets (common + challenging)
            folders = ['helen/testset', 'lfpw/testset', 'ibug']
            xml_file = 'labels_ibug_300W_test.xml'
        
        # Load annotations from XML
        self.load_annotations(xml_file)
        
    def load_annotations(self, xml_file):
        """Load annotations from 300W XML file"""
        xml_path = os.path.join(self.root_dir, xml_file)
        
        if not os.path.exists(xml_path):
            print(f"Warning: {xml_path} not found. Trying to load from folders...")
            self.load_from_folders()
            return
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for image in root.find('images').findall('image'):
            img_path = os.path.join(self.root_dir, image.get('file'))
            
            if not os.path.exists(img_path):
                continue
            
            # Get bounding box
            box = image.find('box')
            if box is None:
                continue
            
            # Get landmarks (68 points)
            landmarks = []
            for part in box.findall('part'):
                x = float(part.get('x'))
                y = float(part.get('y'))
                landmarks.append([x, y])
            
            if len(landmarks) != 68:
                continue
            
            # Get image dimensions
            width = int(image.get('width'))
            height = int(image.get('height'))
            
            # Get bounding box coordinates
            left = float(box.get('left'))
            top = float(box.get('top'))
            box_width = float(box.get('width'))
            box_height = float(box.get('height'))
            
            self.data.append({
                'image_path': img_path,
                'landmarks': np.array(landmarks, dtype=np.float32),
                'bbox': [left, top, box_width, box_height],
                'img_size': [width, height]
            })
        
        print(f"Loaded {len(self.data)} images for {self.split} split")
    
    def load_from_folders(self):
        """Fallback: Load from folder structure with .pts files"""
        folders = {
            'train': ['afw', 'helen/trainset', 'lfpw/trainset'],
            'test': ['helen/testset', 'lfpw/testset', 'ibug']
        }
        
        for folder in folders[self.split]:
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.exists(folder_path):
                continue
            
            for img_name in os.listdir(folder_path):
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                img_path = os.path.join(folder_path, img_name)
                pts_path = img_path.rsplit('.', 1)[0] + '.pts'
                
                if not os.path.exists(pts_path):
                    continue
                
                # Read landmarks from .pts file
                landmarks = self.read_pts_file(pts_path)
                if landmarks is None or len(landmarks) != 68:
                    continue
                
                # Get image size
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    img.close()
                except:
                    continue
                
                self.data.append({
                    'image_path': img_path,
                    'landmarks': landmarks,
                    'bbox': [0, 0, width, height],
                    'img_size': [width, height]
                })
        
        print(f"Loaded {len(self.data)} images from folders for {self.split} split")
    
    def read_pts_file(self, pts_path):
        """Read landmarks from .pts file"""
        try:
            with open(pts_path, 'r') as f:
                lines = f.readlines()
            
            # Find data lines (skip header)
            data_lines = []
            in_data = False
            for line in lines:
                if line.strip() == '{':
                    in_data = True
                    continue
                if line.strip() == '}':
                    break
                if in_data:
                    data_lines.append(line.strip())
            
            # Parse coordinates
            landmarks = []
            for line in data_lines:
                if not line:
                    continue
                coords = line.split()
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    landmarks.append([x, y])
            
            return np.array(landmarks, dtype=np.float32)
        except:
            return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        landmarks = sample['landmarks'].copy()
        
        # Crop to bounding box with padding
        bbox = sample['bbox']
        img_w, img_h = sample['img_size']
        
        # Add padding to bbox
        padding = 0.2
        x, y, w, h = bbox
        x_pad = w * padding
        y_pad = h * padding
        x = max(0, x - x_pad)
        y = max(0, y - y_pad)
        w = min(img_w - x, w + 2 * x_pad)
        h = min(img_h - y, h + 2 * y_pad)
        
        # Crop image
        image = image.crop((int(x), int(y), int(x + w), int(y + h)))
        
        # Adjust landmarks relative to crop
        landmarks[:, 0] -= x
        landmarks[:, 1] -= y
        
        # Resize image to target size
        orig_w, orig_h = image.size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Scale landmarks
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y
        
        # Normalize landmarks to [0, 1]
        landmarks_normalized = landmarks / self.img_size
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.from_numpy(landmarks_normalized).float()


def get_transforms(img_size=112, augment=True):
    """Get data transforms"""
    if augment:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    return transform


def compute_nme(predictions, targets, img_size=112):
    """
    Compute Normalized Mean Error (NME)
    Normalized by inter-ocular distance
    """
    # Convert normalized coordinates back to pixel coordinates
    pred_pts = predictions * img_size
    target_pts = targets * img_size
    
    # Compute inter-ocular distance (between eyes)
    # Left eye: points 36-41, Right eye: points 42-47
    left_eye = target_pts[:, 36:42, :].mean(dim=1)
    right_eye = target_pts[:, 42:48, :].mean(dim=1)
    inter_ocular = torch.norm(left_eye - right_eye, dim=1)
    
    # Compute mean error
    error = torch.norm(pred_pts - target_pts, dim=2).mean(dim=1)
    
    # Normalize by inter-ocular distance
    nme = error / inter_ocular
    
    return nme.mean().item() * 100  # Return as percentage


def compute_accuracy(predictions, targets, img_size=112, threshold=0.05):
    """
    Compute accuracy as percentage of landmarks within threshold
    threshold: normalized distance threshold (default 0.05 = 5% of image size)
    """
    # Compute L2 distance for each landmark
    distances = torch.norm(predictions - targets, dim=2)  # Bx68
    
    # Count landmarks within threshold
    accurate = (distances < threshold).float()
    
    # Average accuracy across all landmarks and batch
    accuracy = accurate.mean().item() * 100
    
    return accuracy


def train_epoch(model, train_loader, optimizer, criterion, device, img_size=112):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_nme = 0.0
    running_acc = 0.0
    
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
        
        # Compute MSE loss
        loss = criterion(pred_landmarks_norm, landmarks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            nme = compute_nme(pred_landmarks_norm, landmarks, img_size)
            acc = compute_accuracy(pred_landmarks_norm, landmarks, img_size)
        
        running_loss += loss.item()
        running_nme += nme
        running_acc += acc
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'NME': running_nme / (pbar.n + 1),
            'Acc': running_acc / (pbar.n + 1)
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_nme = running_nme / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    return epoch_loss, epoch_nme, epoch_acc


def validate(model, val_loader, criterion, device, img_size=112):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_nme = 0.0
    running_acc = 0.0
    
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
            loss = criterion(pred_landmarks_norm, landmarks)
            nme = compute_nme(pred_landmarks_norm, landmarks, img_size)
            acc = compute_accuracy(pred_landmarks_norm, landmarks, img_size)
            
            running_loss += loss.item()
            running_nme += nme
            running_acc += acc
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'NME': running_nme / (pbar.n + 1),
                'Acc': running_acc / (pbar.n + 1)
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_nme = running_nme / len(val_loader)
    epoch_acc = running_acc / len(val_loader)
    return epoch_loss, epoch_nme, epoch_acc


def main():
    # Configuration
    CONFIG = {
        'data_root': './300w',  # Update this to your 300W path
        'img_size': 112,
        'batch_size': 16,
        'num_epochs': 200,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_landmarks',
    }
    
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    print("="*60)
    print("300W Facial Landmark Training")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    print(f"Image size: {CONFIG['img_size']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Optimizer: SGD (lr={CONFIG['learning_rate']}, momentum={CONFIG['momentum']})")
    print(f"Loss: MSE (Mean Squared Error)")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print("="*60)
    
    # Prepare datasets
    train_transform = get_transforms(CONFIG['img_size'], augment=True)
    val_transform = get_transforms(CONFIG['img_size'], augment=False)
    
    train_dataset = W300Dataset(
        CONFIG['data_root'], 
        split='train', 
        transform=train_transform,
        img_size=CONFIG['img_size']
    )
    val_dataset = W300Dataset(
        CONFIG['data_root'], 
        split='test', 
        transform=val_transform,
        img_size=CONFIG['img_size']
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    
    # Data loaders - only use pin_memory if CUDA is available
    use_pin_memory = CONFIG['device'] == 'cuda'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=use_pin_memory
    )
    
    # Create model
    model = FAN(num_landmarks=68)
    model = model.to(CONFIG['device'])
    
    # MSE Loss
    criterion = nn.MSELoss()
    
    # SGD Optimizer with momentum
    optimizer = optim.SGD(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_nme = float('inf')
    train_losses, train_nmes, train_accs = [], [], []
    val_losses, val_nmes, val_accs = [], [], []
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_nme, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, CONFIG['device'], CONFIG['img_size']
        )
        train_losses.append(train_loss)
        train_nmes.append(train_nme)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_nme, val_acc = validate(
            model, val_loader, criterion, CONFIG['device'], CONFIG['img_size']
        )
        val_losses.append(val_loss)
        val_nmes.append(val_nme)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_nme)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train - Loss: {train_loss:.4f} | NME: {train_nme:.2f}% | Accuracy: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f} | NME: {val_nme:.2f}% | Accuracy: {val_acc:.2f}%")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_nme < best_nme:
            best_nme = val_nme
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nme': val_nme,
                'accuracy': val_acc,
            }, os.path.join(CONFIG['save_dir'], 'best_fan_model.pth'))
            print(f"✓ Best model saved! NME: {val_nme:.2f}% | Accuracy: {val_acc:.2f}%")
    
    print("\n" + "="*60)
    print(f"Training completed! Best NME: {best_nme:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()