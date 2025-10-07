import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================
# FAN Model Architecture (for loading only)
# ============================================

class FAN(nn.Module):
    """Face Alignment Network - same as Phase 1"""
    def __init__(self, num_landmarks=68):
        super(FAN, self).__init__()
        self.num_landmarks = num_landmarks
        
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
        
        return features, landmarks, heatmaps


# ============================================
# Landmark Detector Wrapper
# ============================================

class LandmarkDetector:
    """Loads trained FAN model and detects landmarks"""
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        print(f"Loading landmark detection model from: {model_path}")
        
        # Create model
        self.model = FAN(num_landmarks=68)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  NME: {checkpoint.get('nme', 'N/A')}")
            print(f"  Accuracy: {checkpoint.get('accuracy', 'N/A')}")
        else:
            raise ValueError("Checkpoint format not recognized.")
        
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_landmarks(self, image):
        """
        Detect 68 facial landmarks from an image
        Args:
            image: PIL Image or numpy array (RGB)
        Returns:
            landmarks: numpy array (68, 2) with x,y coordinates
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, landmarks, heatmaps = self.model(img_tensor)
        
        h, w = heatmaps.size(2), heatmaps.size(3)
        landmarks = landmarks.cpu().numpy()[0]
        
        # Scale to 112x112
        landmarks[:, 0] = landmarks[:, 0] * 112 / (w - 1)
        landmarks[:, 1] = landmarks[:, 1] * 112 / (h - 1)
        
        return landmarks


# ============================================
# RAF-DB Dataset
# ============================================

class RAFDBDataset(Dataset):
    """RAF-DB Dataset loader"""
    def __init__(self, root_dir, split='train', landmark_detector=None, 
                 img_size=112, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.landmark_detector = landmark_detector
        self.transform = transform
        
        self.img_dir = os.path.join(root_dir, split)
        self.samples = []
        self._load_samples()
        
        self.emotion_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
        
        print(f"✓ Loaded {len(self.samples)} samples from {split} set")
    
    def _load_samples(self):
        """Load all image paths from folder structure"""
        for class_folder in sorted(os.listdir(self.img_dir)):
            class_path = os.path.join(self.img_dir, class_folder)
            
            if not os.path.isdir(class_path):
                continue
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    label = int(class_folder) - 1
                    
                    self.samples.append({
                        'img_path': img_path,
                        'img_name': img_name,
                        'label': label,
                        'class_folder': class_folder
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img_path = sample['img_path']
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        landmarks = None
        if self.landmark_detector is not None:
            try:
                landmarks = self.landmark_detector.detect_landmarks(image)
            except Exception as e:
                print(f"Warning: Failed landmarks for {sample['img_name']}: {e}")
                landmarks = np.zeros((68, 2), dtype=np.float32)
        
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size), 
                          interpolation=cv2.INTER_LINEAR)
        
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': sample['label'],
            'landmarks': landmarks if landmarks is not None else np.zeros((68, 2)),
            'img_name': sample['img_name']
        }


# ============================================
# Process RAF-DB with Landmarks
# ============================================

def prepare_raf_db(dataset_root, landmark_model_path, output_dir, device):
    """Process RAF-DB dataset and extract landmarks"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASE 2: RAF-DB DATA PREPARATION")
    print("="*60)
    
    # Load landmark detector
    landmark_detector = LandmarkDetector(landmark_model_path, device=device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Process both splits
    for split in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} set...")
        print(f"{'='*60}")
        
        dataset = RAFDBDataset(
            root_dir=dataset_root,
            split=split,
            landmark_detector=landmark_detector,
            transform=transform
        )
        
        landmarks_list = []
        labels_list = []
        img_names_list = []
        
        for i in tqdm(range(len(dataset)), desc=f"Processing {split}"):
            try:
                sample = dataset[i]
                
                # Save image
                img_tensor = sample['image']
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                label = sample['label']
                class_folder = str(label + 1)
                class_output_dir = os.path.join(output_dir, split, class_folder)
                os.makedirs(class_output_dir, exist_ok=True)
                
                save_path = os.path.join(class_output_dir, sample['img_name'])
                cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                
                # Store metadata
                landmarks_list.append(sample['landmarks'])
                labels_list.append(sample['label'])
                img_names_list.append(f"{class_folder}/{sample['img_name']}")
                
            except Exception as e:
                print(f"\nError processing sample {i}: {e}")
                continue
        
        # Save arrays
        landmarks_array = np.array(landmarks_list)
        labels_array = np.array(labels_list)
        
        np.save(os.path.join(output_dir, f'{split}_landmarks.npy'), landmarks_array)
        np.save(os.path.join(output_dir, f'{split}_labels.npy'), labels_array)
        
        with open(os.path.join(output_dir, f'{split}_img_names.txt'), 'w') as f:
            f.write('\n'.join(img_names_list))
        
        print(f"\n✓ {split} set complete: {len(labels_list)} samples")
        print(f"  Landmarks: {landmarks_array.shape}")
        print(f"  Labels: {labels_array.shape}")


# ============================================
# Verify Processed Data
# ============================================

def verify_data(output_dir):
    """Verify the processed dataset"""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    emotion_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
    
    for split in ['train', 'test']:
        landmarks = np.load(os.path.join(output_dir, f'{split}_landmarks.npy'))
        labels = np.load(os.path.join(output_dir, f'{split}_labels.npy'))
        
        print(f"\n{split.upper()} SET:")
        print(f"  Total samples: {len(labels)}")
        print(f"  Landmarks shape: {landmarks.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"\n  Class distribution:")
        
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            percentage = (count / len(labels)) * 100
            print(f"    {label} - {emotion_names[label]:8s}: {count:5d} ({percentage:5.2f}%)")
        
        valid = np.all(landmarks != 0, axis=(1, 2))
        print(f"\n  Valid landmarks: {valid.sum()} / {len(labels)}")


# ============================================
# Visualize Samples
# ============================================

def visualize_samples(output_dir, num_samples=5):
    """Visualize processed samples with landmarks"""
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    landmarks = np.load(os.path.join(output_dir, 'train_landmarks.npy'))
    labels = np.load(os.path.join(output_dir, 'train_labels.npy'))
    
    with open(os.path.join(output_dir, 'train_img_names.txt'), 'r') as f:
        img_names = f.read().splitlines()
    
    emotion_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
    
    indices = np.random.choice(len(labels), min(num_samples, len(labels)), replace=False)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(4*len(indices), 4))
    if len(indices) == 1:
        axes = [axes]
    
    for idx, ax in zip(indices, axes):
        img_path = os.path.join(output_dir, 'train', img_names[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img)
        
        lm = landmarks[idx]
        if not np.all(lm == 0):
            ax.scatter(lm[:, 0], lm[:, 1], c='red', s=10, alpha=0.7)
        
        emotion = emotion_names[labels[idx]]
        ax.set_title(f"{emotion}\n{img_names[idx].split('/')[-1]}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sample_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    # Configuration
    DATASET_ROOT = 'archive/DATASET'
    LANDMARK_MODEL = 'checkpoints_landmarks/best_fan_model.pth'
    OUTPUT_DIR = 'processed_raf_db'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}\n")
    
    # Process data
    prepare_raf_db(
        dataset_root=DATASET_ROOT,
        landmark_model_path=LANDMARK_MODEL,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )
    
    # Verify
    verify_data(OUTPUT_DIR)
    
    # Visualize
    try:
        visualize_samples(OUTPUT_DIR, num_samples=5)
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
    
    print("\n" + "="*60)
    print("✓ PHASE 2 COMPLETE!")
    print("="*60)
    print(f"\nOutput: {OUTPUT_DIR}/")
    print("\nReady for Phase 3: FER Model Training")
    print("="*60)