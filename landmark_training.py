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
import cv2
import face_alignment

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
        self.linear = nn.Linear(512, num_landmarks * 2)
        
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
        return out.view(-1, 68, 2)

# ======================= PFA Teacher Network =======================
class PFANetwork(nn.Module):
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

# ======================= Face Alignment Landmark Extractor =======================
class FaceAlignmentExtractor:
    """Extract 68-point facial landmarks using face_alignment library"""
    def __init__(self, device='cuda'):
        """
        Initialize face_alignment detector
        
        Args:
            device: 'cuda' or 'cpu'
        """
        try:
            # Use 2D landmarks with the best available model
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=device,
                flip_input=False
            )
            print(f"Loaded face_alignment on {device}")
        except Exception as e:
            print(f"Failed to load on {device}, falling back to CPU")
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device='cpu',
                flip_input=False
            )
            print("Loaded face_alignment on CPU")
    
    def extract_landmarks(self, image_path, target_size=(112, 112)):
        """
        Extract 68 facial landmarks from image
        
        Args:
            image_path: Path to image file
            target_size: Target image size (width, height)
            
        Returns:
            landmarks: numpy array of shape (68, 2) with normalized coordinates
            success: boolean indicating if face was detected
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None, False
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_h, original_w = img_rgb.shape[:2]
            
            # Detect landmarks
            preds = self.fa.get_landmarks(img_rgb)
            
            if preds is None or len(preds) == 0:
                return None, False
            
            # Get first face landmarks (68 points)
            landmarks = preds[0]
            
            # Normalize landmarks to target size
            scale_x = target_size[0] / original_w
            scale_y = target_size[1] / original_h
            
            landmarks[:, 0] *= scale_x
            landmarks[:, 1] *= scale_y
            
            return landmarks.astype(np.float32), True
            
        except Exception as e:
            return None, False

# ======================= Dataset with Face Alignment Landmarks =======================
class RAFDBLandmarkDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, 
                 device='cuda', cache_landmarks=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.cache_landmarks = cache_landmarks
        self.data = []
        
        # Initialize face_alignment extractor
        print("\nInitializing face_alignment landmark extractor...")
        try:
            self.landmark_extractor = FaceAlignmentExtractor(device=device)
        except Exception as e:
            print(f"\nError: {e}")
            print("\nTo install face_alignment:")
            print("  pip install face-alignment")
            raise
        
        # Dataset structure
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
            raise ValueError(f"Could not find {split} directory in {root_dir}")
        
        self.img_dir = img_dir
        print(f"Using directory: {img_dir}")
        
        # Cache file for landmarks
        cache_file = os.path.join(root_dir, f'landmarks_cache_facealign_{split}.npy')
        
        # Load cached landmarks if available
        if cache_landmarks and os.path.exists(cache_file):
            print(f"Loading cached landmarks from {cache_file}...")
            cache_data = np.load(cache_file, allow_pickle=True).item()
            self.data = cache_data['data']
            print(f"Loaded {len(self.data)} cached samples")
            return
        
        # Extract landmarks for all images
        emotion_dirs = ['1', '2', '3', '4', '5', '6', '7']
        failed_extractions = 0
        
        print("\nExtracting landmarks from images...")
        for emotion_class in emotion_dirs:
            class_dir = os.path.join(img_dir, emotion_class)
            if not os.path.exists(class_dir):
                continue
                
            images = [f for f in os.listdir(class_dir) 
                     if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'))]
            
            print(f"Processing class {emotion_class}: {len(images)} images")
            
            for img_name in tqdm(images, desc=f"Class {emotion_class}"):
                img_path = os.path.join(class_dir, img_name)
                
                # Extract landmarks using face_alignment
                landmarks, success = self.landmark_extractor.extract_landmarks(img_path)
                
                if success:
                    rel_path = os.path.join(emotion_class, img_name)
                    self.data.append((rel_path, landmarks, int(emotion_class)))
                else:
                    failed_extractions += 1
        
        print(f"\nSuccessfully extracted landmarks for {len(self.data)} images")
        if failed_extractions > 0:
            print(f"Failed to detect faces in {failed_extractions} images (skipped)")
        
        # Cache landmarks for future use
        if cache_landmarks and len(self.data) > 0:
            print(f"Saving landmarks cache to {cache_file}...")
            np.save(cache_file, {'data': self.data})
            print("Cache saved")
    
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
            
            return image, landmarks
        except Exception as e:
            print(f"Error loading {rel_path}: {e}")
            # Return a dummy sample
            dummy_img = torch.zeros(3, 112, 112)
            dummy_landmarks = torch.zeros(68, 2)
            return dummy_img, dummy_landmarks

# ======================= Accuracy Metrics =======================
class LandmarkAccuracyMetrics:
    """Calculate accuracy metrics for facial landmark detection"""
    
    @staticmethod
    def calculate_nme(predictions, targets, normalization='inter_ocular'):
        """Calculate Normalized Mean Error (NME)"""
        distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=2))
        
        if normalization == 'inter_ocular':
            # Distance between outer eye corners (landmarks 36 and 45)
            left_eye = targets[:, 36, :]
            right_eye = targets[:, 45, :]
            norm_factor = torch.sqrt(torch.sum((left_eye - right_eye) ** 2, dim=1))
        else:
            # Bounding box diagonal
            bbox_min = torch.min(targets, dim=1)[0]
            bbox_max = torch.max(targets, dim=1)[0]
            norm_factor = torch.sqrt(torch.sum((bbox_max - bbox_min) ** 2, dim=1))
        
        norm_factor = torch.clamp(norm_factor, min=1e-6)
        nme = torch.mean(distances, dim=1) / norm_factor
        return torch.mean(nme).item()
    
    @staticmethod
    def calculate_auc(predictions, targets, threshold=0.08):
        """Calculate Area Under Curve (AUC)"""
        distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=2))
        
        left_eye = targets[:, 36, :]
        right_eye = targets[:, 45, :]
        norm_factor = torch.sqrt(torch.sum((left_eye - right_eye) ** 2, dim=1))
        norm_factor = torch.clamp(norm_factor, min=1e-6)
        
        nme_per_sample = torch.mean(distances, dim=1) / norm_factor
        auc = (nme_per_sample < threshold).float().mean().item() * 100
        return auc
    
    @staticmethod
    def calculate_failure_rate(predictions, targets, threshold=0.1):
        """Calculate failure rate"""
        distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=2))
        
        left_eye = targets[:, 36, :]
        right_eye = targets[:, 45, :]
        norm_factor = torch.sqrt(torch.sum((left_eye - right_eye) ** 2, dim=1))
        norm_factor = torch.clamp(norm_factor, min=1e-6)
        
        nme_per_sample = torch.mean(distances, dim=1) / norm_factor
        failure_rate = (nme_per_sample > threshold).float().mean().item() * 100
        return failure_rate
    
    @staticmethod
    def calculate_accuracy(predictions, targets, pixel_threshold=5.0):
        """Calculate simple accuracy: percentage of landmarks within pixel threshold"""
        distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=2))
        correct_landmarks = (distances < pixel_threshold).float()
        accuracy = torch.mean(correct_landmarks).item() * 100
        return accuracy

# ======================= Knowledge Distillation Loss =======================
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_output, teacher_output, target):
        hard_loss = self.mse_loss(student_output, target)
        soft_loss = self.mse_loss(
            student_output / self.temperature,
            teacher_output.detach() / self.temperature
        )
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss, hard_loss, soft_loss

# ======================= Training Pipeline =======================
class FaceAlignmentTrainer:
    def __init__(self, student_model, teacher_model, device='cuda'):
        self.device = device
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.teacher.eval()
        
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
            'train_soft_loss': [],
            'train_nme': [],
            'val_nme': [],
            'train_auc': [],
            'val_auc': [],
            'val_failure_rate': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        self.metrics = LandmarkAccuracyMetrics()
    
    def train_epoch(self, train_loader):
        self.student.train()
        running_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0
        running_nme = 0.0
        running_auc = 0.0
        running_accuracy = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            student_output = self.student(images)
            
            with torch.no_grad():
                teacher_output = self.teacher(images)
            
            loss, hard_loss, soft_loss = self.criterion(
                student_output, teacher_output, targets
            )
            
            with torch.no_grad():
                nme = self.metrics.calculate_nme(student_output, targets)
                auc = self.metrics.calculate_auc(student_output, targets)
                accuracy = self.metrics.calculate_accuracy(student_output, targets, pixel_threshold=5.0)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            running_hard_loss += hard_loss.item()
            running_soft_loss += soft_loss.item()
            running_nme += nme
            running_auc += auc
            running_accuracy += accuracy
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.1f}%',
                'NME': f'{nme:.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_hard_loss = running_hard_loss / len(train_loader)
        epoch_soft_loss = running_soft_loss / len(train_loader)
        epoch_nme = running_nme / len(train_loader)
        epoch_auc = running_auc / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)
        
        return epoch_loss, epoch_hard_loss, epoch_soft_loss, epoch_nme, epoch_auc, epoch_accuracy
    
    def validate(self, val_loader):
        self.student.eval()
        running_loss = 0.0
        running_nme = 0.0
        running_auc = 0.0
        running_failure_rate = 0.0
        running_accuracy = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                student_output = self.student(images)
                loss = nn.MSELoss()(student_output, targets)
                
                nme = self.metrics.calculate_nme(student_output, targets)
                auc = self.metrics.calculate_auc(student_output, targets, threshold=0.08)
                failure_rate = self.metrics.calculate_failure_rate(student_output, targets, threshold=0.1)
                accuracy = self.metrics.calculate_accuracy(student_output, targets, pixel_threshold=5.0)
                
                running_loss += loss.item()
                running_nme += nme
                running_auc += auc
                running_failure_rate += failure_rate
                running_accuracy += accuracy
        
        avg_loss = running_loss / len(val_loader)
        avg_nme = running_nme / len(val_loader)
        avg_auc = running_auc / len(val_loader)
        avg_failure_rate = running_failure_rate / len(val_loader)
        avg_accuracy = running_accuracy / len(val_loader)
        
        return avg_loss, avg_nme, avg_auc, avg_failure_rate, avg_accuracy
    
    def train(self, train_loader, val_loader, epochs=200, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        best_val_accuracy = 0.0
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Student parameters: {sum(p.numel() for p in self.student.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 70)
            
            train_loss, hard_loss, soft_loss, train_nme, train_auc, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_hard_loss'].append(hard_loss)
            self.history['train_soft_loss'].append(soft_loss)
            self.history['train_nme'].append(train_nme)
            self.history['train_auc'].append(train_auc)
            self.history['train_accuracy'].append(train_acc)
            
            val_loss, val_nme, val_auc, val_failure_rate, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_nme'].append(val_nme)
            self.history['val_auc'].append(val_auc)
            self.history['val_failure_rate'].append(val_failure_rate)
            self.history['val_accuracy'].append(val_acc)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1} RESULTS")
            print(f"{'='*70}")
            print(f"TRAINING:")
            print(f"  Loss: {train_loss:.6f} (Hard: {hard_loss:.6f}, Soft: {soft_loss:.6f})")
            print(f"  Accuracy: {train_acc:.2f}% | NME: {train_nme:.6f} | AUC@8%: {train_auc:.2f}%")
            print(f"\nVALIDATION:")
            print(f"  Loss: {val_loss:.6f}")
            print(f"  Accuracy: {val_acc:.2f}% | NME: {val_nme:.6f} | AUC@8%: {val_auc:.2f}%")
            print(f"  Failure@10%: {val_failure_rate:.2f}%")
            print(f"\nLearning Rate: {current_lr:.6f}")
            print(f"{'='*70}")
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_nme': val_nme,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"\nNEW BEST MODEL! Accuracy: {val_acc:.2f}% (NME: {val_nme:.6f})")
            
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
        fig, axes = plt.subplots(3, 3, figsize=(20, 12))
        
        # Plot 1: Accuracy
        axes[0, 0].plot(self.history['train_accuracy'], label='Train', linewidth=2.5, color='blue')
        axes[0, 0].plot(self.history['val_accuracy'], label='Val', linewidth=2.5, color='red')
        axes[0, 0].set_xlabel('Epoch', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[0, 0].set_title('Landmark Accuracy (within 5px)', fontweight='bold', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 100])
        
        # Plot 2: Loss
        axes[0, 1].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training and Validation Loss', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Hard vs Soft Loss
        axes[0, 2].plot(self.history['train_hard_loss'], label='Hard Loss', linewidth=2)
        axes[0, 2].plot(self.history['train_soft_loss'], label='Soft Loss', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Distillation Loss Components', fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: NME
        axes[1, 0].plot(self.history['train_nme'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history['val_nme'], label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('NME')
        axes[1, 0].set_title('Normalized Mean Error', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: AUC
        axes[1, 1].plot(self.history['train_auc'], label='Train', linewidth=2, color='green')
        axes[1, 1].plot(self.history['val_auc'], label='Val', linewidth=2, color='darkgreen')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC (%)')
        axes[1, 1].set_title('Detection Rate @ 8%', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 100])
        
        # Plot 6: Failure Rate
        axes[1, 2].plot(self.history['val_failure_rate'], label='Failure Rate', linewidth=2, color='red')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Failure Rate (%)')
        axes[1, 2].set_title('Validation Failure Rate @ 10%', fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Plot 7: Accuracy Filled
        axes[2, 0].fill_between(range(len(self.history['train_accuracy'])), 
                                self.history['train_accuracy'], alpha=0.3, color='blue')
        axes[2, 0].fill_between(range(len(self.history['val_accuracy'])), 
                                self.history['val_accuracy'], alpha=0.3, color='red')
        axes[2, 0].plot(self.history['train_accuracy'], linewidth=2, color='blue', label='Train')
        axes[2, 0].plot(self.history['val_accuracy'], linewidth=2, color='red', label='Val')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Accuracy (%)')
        axes[2, 0].set_title('Accuracy Progress', fontweight='bold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_ylim([0, 100])
        
        # Plot 8: Log Loss
        axes[2, 1].plot(self.history['train_loss'], linewidth=2, label='Train')
        axes[2, 1].plot(self.history['val_loss'], linewidth=2, label='Val')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Loss (log scale)')
        axes[2, 1].set_title('Training Progress (Log Scale)', fontweight='bold')
        axes[2, 1].set_yscale('log')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Plot 9: Summary Stats
        axes[2, 2].axis('off')
        summary = f"""
FINAL TRAINING SUMMARY
{'='*35}

ACCURACY METRICS:
  Best Val Acc:     {max(self.history['val_accuracy']):.2f}%
  Final Val Acc:    {self.history['val_accuracy'][-1]:.2f}%
  Best Train Acc:   {max(self.history['train_accuracy']):.2f}%

ERROR METRICS:
  Best Val NME:     {min(self.history['val_nme']):.6f}
  Final Val NME:    {self.history['val_nme'][-1]:.6f}
  Best AUC@8%:      {max(self.history['val_auc']):.2f}%
  Final Fail@10%:   {self.history['val_failure_rate'][-1]:.2f}%

TRAINING INFO:
  Total Epochs:     {len(self.history['train_loss'])}
  Best Loss:        {min(self.history['val_loss']):.6f}
        """
        axes[2, 2].text(0.1, 0.95, summary, transform=axes[2, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*70}")
        print("FINAL TRAINING STATISTICS")
        print(f"{'='*70}")
        print(f"Best Validation Accuracy: {max(self.history['val_accuracy']):.2f}%")
        print(f"Final Validation Accuracy: {self.history['val_accuracy'][-1]:.2f}%")
        print(f"Best Validation NME: {min(self.history['val_nme']):.6f}")
        print(f"Best Validation AUC@8%: {max(self.history['val_auc']):.2f}%")
        print(f"Final Failure Rate@10%: {self.history['val_failure_rate'][-1]:.2f}%")
        print(f"\nTraining history plot saved to: {save_path}")
        print(f"{'='*70}")

# ======================= Main Training Script =======================
def main():
    BATCH_SIZE = 16
    EPOCHS = 200
    DATA_ROOT = './archive'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("="*60)
    
    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import face_alignment
        import cv2
        print("face_alignment and OpenCV installed")
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall required packages:")
        print("  pip install face-alignment opencv-python")
        return
    
    # Check dataset
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Dataset directory not found: {DATA_ROOT}")
        return
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading datasets with face_alignment landmark extraction...")
    print("="*60)
    
    train_dataset = RAFDBLandmarkDataset(
        DATA_ROOT, 
        transform=transform, 
        train=True,
        device=device,
        cache_landmarks=True
    )
    
    val_dataset = RAFDBLandmarkDataset(
        DATA_ROOT, 
        transform=transform, 
        train=False,
        device=device,
        cache_landmarks=True
    )
    
    print(f"\nDataset loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\nError: No valid samples with face detections!")
        return
    
    # Create data loaders
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # Initialize models
    print("\nInitializing models...")
    student_model = MobileFaceNet(num_landmarks=68)
    teacher_model = PFANetwork(num_landmarks=68)
    
    # Load pretrained teacher if available
    teacher_checkpoint = 'pfa_pretrained.pth'
    if os.path.exists(teacher_checkpoint):
        print(f"Loading pretrained teacher from {teacher_checkpoint}")
        teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
    else:
        print("Note: No pretrained teacher model found. Training from scratch.")
    
    # Initialize trainer
    trainer = FaceAlignmentTrainer(student_model, teacher_model, device=device)
    
    # Train
    print("\n" + "="*60)
    print("Starting training with face_alignment landmarks (68 points)")
    print("="*60)
    
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Plot training history
    trainer.plot_history()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Best model: checkpoints/best_model.pth")
    print(f"Training history: training_history.png")
    print("="*60)

if __name__ == '__main__':
    main()