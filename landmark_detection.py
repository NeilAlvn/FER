import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ======================= MobileFaceNet Architecture =======================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    
    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_c, groups, kernel=(1, 1), stride=(1, 1), padding=(0, 0)),
            ConvBlock(groups, groups, kernel=kernel, stride=stride, padding=padding, groups=groups),
            nn.Conv2d(groups, out_c, (1, 1), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.residual = residual
    
    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            return out + x
        return out

class MobileFaceNet(nn.Module):
    def __init__(self, num_landmarks=68):
        super(MobileFaceNet, self).__init__()
        
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = DepthWise(64, 64, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        
        self.conv_4 = nn.Sequential(
            DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256),
            DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256),
            DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256),
            DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        )
        
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_5 = nn.Sequential(
            DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256),
            DepthWise(128, 128, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        )
        
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        
        self.conv_6_flatten = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_landmarks * 2)  # x, y coordinates
        
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
        out = self.conv_6_flatten(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out.view(-1, 68, 2)  # Reshape to (batch, landmarks, coordinates)

# ======================= PFA Teacher Network (Pretrained) =======================
class PFANetwork(nn.Module):
    """High-accuracy teacher network for knowledge distillation"""
    def __init__(self, num_landmarks=68):
        super(PFANetwork, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            self._make_layer(64, 128, 3),
            self._make_layer(128, 256, 4, stride=2),
            self._make_layer(256, 512, 6, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(512, num_landmarks * 2)
    
    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_c, out_c, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(out_c, out_c, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.view(-1, 68, 2)

# ======================= Dataset =======================
class RAFDBLandmarkDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.data = []
        
        # Your dataset structure: archive/DATASET/train/ and archive/DATASET/test/
        split = 'train' if train else 'test'
        
        # Try multiple possible paths
        possible_base_paths = [
            os.path.join(root_dir, 'archive', 'DATASET', split),
            os.path.join(root_dir, 'DATASET', split),
            os.path.join(root_dir, split)
        ]
        
        img_dir = None
        for base_path in possible_base_paths:
            if os.path.exists(base_path):
                img_dir = base_path
                break
        
        if img_dir is None:
            print(f"Error: Could not find {split} directory in {root_dir}")
            print(f"Please check your dataset path.")
            return
        
        self.img_dir = img_dir
        print(f"Using directory: {img_dir}")
        
        # Load images from subdirectories (1, 2, 3, 4, 5, 6, 7)
        # These represent emotion classes in RAF-DB
        emotion_dirs = ['1', '2', '3', '4', '5', '6', '7']
        
        for emotion_class in emotion_dirs:
            class_dir = os.path.join(img_dir, emotion_class)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]
                
                print(f"  Found {len(images)} images in class {emotion_class}")
                
                for img_name in images:
                    # Generate synthetic landmarks (68-point facial landmarks)
                    # In production, use actual landmark annotations if available
                    landmarks = self._generate_synthetic_landmarks()
                    
                    # Store relative path from img_dir
                    rel_path = os.path.join(emotion_class, img_name)
                    self.data.append((rel_path, landmarks, int(emotion_class)))
        
        if len(self.data) == 0:
            print(f"Warning: No images found for {'training' if train else 'validation'}")
            print(f"Checked directory: {img_dir}")
            if os.path.exists(img_dir):
                print(f"Subdirectories found: {os.listdir(img_dir)}")
        else:
            print(f"Total {split} samples loaded: {len(self.data)}")
    
    def _generate_synthetic_landmarks(self):
        """Generate synthetic 68 facial landmarks for demonstration purposes.
        In production, use actual landmark annotations."""
        # Standard 68-point facial landmark positions (normalized 0-1)
        landmarks = np.array([
            # Jawline (0-16)
            [0.2, 0.3], [0.22, 0.35], [0.24, 0.4], [0.26, 0.45], [0.28, 0.5],
            [0.3, 0.55], [0.35, 0.6], [0.4, 0.65], [0.5, 0.68],
            [0.6, 0.65], [0.65, 0.6], [0.7, 0.55], [0.72, 0.5],
            [0.74, 0.45], [0.76, 0.4], [0.78, 0.35], [0.8, 0.3],
            # Right eyebrow (17-21)
            [0.3, 0.25], [0.35, 0.23], [0.4, 0.22], [0.45, 0.23], [0.48, 0.25],
            # Left eyebrow (22-26)
            [0.52, 0.25], [0.55, 0.23], [0.6, 0.22], [0.65, 0.23], [0.7, 0.25],
            # Nose (27-35)
            [0.5, 0.3], [0.5, 0.35], [0.5, 0.4], [0.5, 0.45],
            [0.45, 0.47], [0.475, 0.48], [0.5, 0.49], [0.525, 0.48], [0.55, 0.47],
            # Right eye (36-41)
            [0.37, 0.32], [0.4, 0.31], [0.43, 0.31], [0.46, 0.32],
            [0.43, 0.33], [0.4, 0.33],
            # Left eye (42-47)
            [0.54, 0.32], [0.57, 0.31], [0.6, 0.31], [0.63, 0.32],
            [0.6, 0.33], [0.57, 0.33],
            # Mouth outer (48-59)
            [0.4, 0.55], [0.43, 0.54], [0.46, 0.53], [0.5, 0.54],
            [0.54, 0.53], [0.57, 0.54], [0.6, 0.55],
            [0.57, 0.57], [0.54, 0.58], [0.5, 0.59], [0.46, 0.58], [0.43, 0.57],
            # Mouth inner (60-67)
            [0.43, 0.56], [0.46, 0.555], [0.5, 0.56], [0.54, 0.555],
            [0.57, 0.56], [0.54, 0.57], [0.5, 0.575], [0.46, 0.57]
        ])
        
        # Add some randomness
        landmarks += np.random.randn(68, 2) * 0.02
        
        # Scale to image size (assuming 112x112)
        landmarks = landmarks * 112
        
        return landmarks.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rel_path, landmarks, emotion_label = self.data[idx]
        img_path = os.path.join(self.img_dir, rel_path)
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            landmarks = torch.FloatTensor(landmarks)
            
            # Ensure landmarks are in correct format
            if landmarks.shape[0] != 68:
                # Pad or truncate to 68 landmarks
                if landmarks.shape[0] < 68:
                    padding = torch.zeros(68 - landmarks.shape[0], 2)
                    landmarks = torch.cat([landmarks, padding], dim=0)
                else:
                    landmarks = landmarks[:68]
            
            return image, landmarks
        except Exception as e:
            print(f"Error loading {rel_path}: {e}")
            # Return a dummy sample
            dummy_img = torch.zeros(3, 112, 112)
            dummy_landmarks = torch.zeros(68, 2)
            return dummy_img, dummy_landmarks

# ======================= Knowledge Distillation Loss =======================
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_output, teacher_output, target):
        # Hard target loss (MSE with ground truth)
        hard_loss = self.mse_loss(student_output, target)
        
        # Soft target loss (MSE with teacher predictions)
        soft_loss = self.mse_loss(
            student_output / self.temperature,
            teacher_output.detach() / self.temperature
        )
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, hard_loss, soft_loss

# ======================= Training Pipeline =======================
class FaceAlignmentTrainer:
    def __init__(self, student_model, teacher_model, device='cuda'):
        self.device = device
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.teacher.eval()  # Teacher is always in eval mode
        
        # Freeze teacher network
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.criterion = DistillationLoss(alpha=0.5, temperature=3.0)
        self.optimizer = optim.SGD(
            self.student.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[60, 120, 160],
            gamma=0.1
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_hard_loss': [],
            'train_soft_loss': []
        }
    
    def train_epoch(self, train_loader):
        self.student.train()
        running_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get student predictions
            student_output = self.student(images)
            
            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_output = self.teacher(images)
            
            # Calculate loss
            loss, hard_loss, soft_loss = self.criterion(
                student_output, teacher_output, targets
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            running_hard_loss += hard_loss.item()
            running_soft_loss += soft_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hard': f'{hard_loss.item():.4f}',
                'soft': f'{soft_loss.item():.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_hard_loss = running_hard_loss / len(train_loader)
        epoch_soft_loss = running_soft_loss / len(train_loader)
        
        return epoch_loss, epoch_hard_loss, epoch_soft_loss
    
    def validate(self, val_loader):
        self.student.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                student_output = self.student(images)
                loss = nn.MSELoss()(student_output, targets)
                running_loss += loss.item()
        
        return running_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=200, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Student parameters: {sum(p.numel() for p in self.student.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, hard_loss, soft_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_hard_loss'].append(hard_loss)
            self.history['train_soft_loss'].append(soft_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.6f} (Hard: {hard_loss:.6f}, Soft: {soft_loss:.6f})")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ Best model saved!")
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print("\nTraining completed!")
        return self.history
    
    def plot_history(self, save_path='training_history.png'):
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Total Loss
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Hard vs Soft Loss
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_hard_loss'], label='Hard Loss')
        plt.plot(self.history['train_soft_loss'], label='Soft Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Distillation Loss Components')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Learning Progress
        plt.subplot(1, 3, 3)
        plt.plot(self.history['train_loss'], label='Train')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Progress (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

# ======================= Main Training Script =======================
def main():
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 200
    # Update this to your actual path
    DATA_ROOT = './archive'  # or './archive/DATASET' depending on your structure
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Dataset directory not found: {DATA_ROOT}")
        print(f"\nPlease update DATA_ROOT to point to your dataset location.")
        print("\nYour structure should be:")
        print("archive/")
        print("  └── DATASET/")
        print("      ├── train/")
        print("      │   ├── 1/ (Surprise)")
        print("      │   ├── 2/ (Fear)")
        print("      │   ├── 3/ (Disgust)")
        print("      │   ├── 4/ (Happiness)")
        print("      │   ├── 5/ (Sadness)")
        print("      │   ├── 6/ (Anger)")
        print("      │   └── 7/ (Neutral)")
        print("      └── test/")
        print("          └── (same structure)")
        return
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = RAFDBLandmarkDataset(DATA_ROOT, transform=transform, train=True)
    val_dataset = RAFDBLandmarkDataset(DATA_ROOT, transform=transform, train=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\nError: No samples found in dataset!")
        print("Please check:")
        print("1. Dataset path is correct")
        print("2. Images exist in the directory")
        print("3. Image files have correct extensions (.jpg, .png, .jpeg)")
        return
    
    # Adjust num_workers based on platform
    num_workers = 0 if os.name == 'nt' else 4  # Use 0 for Windows, 4 for Unix
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize models
    print("\nInitializing models...")
    student_model = MobileFaceNet(num_landmarks=68)
    teacher_model = PFANetwork(num_landmarks=68)
    
    # Load pretrained teacher model (if available)
    teacher_checkpoint = 'pfa_pretrained.pth'
    if os.path.exists(teacher_checkpoint):
        print(f"Loading pretrained teacher from {teacher_checkpoint}")
        teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
    else:
        print("Note: No pretrained teacher model found. Training from scratch.")
        print("For better results, provide a pretrained PFA teacher model.")
    
    # Initialize trainer
    trainer = FaceAlignmentTrainer(student_model, teacher_model, device=device)
    
    # Train
    print(f"\nStarting training with:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Learning rate: 0.01")
    print(f"  - Optimizer: SGD (momentum=0.9)")
    print(f"  - Device: {device}")
    print("=" * 50)
    
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Plot training history
    trainer.plot_history()
    
    print("\n" + "=" * 50)
    print("Training pipeline completed successfully!")
    print(f"Best model saved to: checkpoints/best_model.pth")
    print(f"Training history saved to: training_history.png")

if __name__ == '__main__':
    main()