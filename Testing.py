import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image


class FAN(nn.Module):
    """Face Alignment Network for landmark detection"""
    def __init__(self, num_landmarks=68):
        super(FAN, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Using MobileNetV2 as backbone for FAN
        mobilenet = models.mobilenet_v2(weights=None)
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


def load_model(model_path, device='cuda'):
    """Load the trained FAN model"""
    model = FAN(num_landmarks=68)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Model NME: {checkpoint.get('nme', 'N/A')}")
    print(f"Model Accuracy: {checkpoint.get('accuracy', 'N/A')}")
    return model


def get_transform():
    """Get preprocessing transform"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    """Draw facial landmarks on image with connections"""
    landmarks = landmarks.astype(int)
    
    # Define facial landmark connections
    connections = [
        # Jaw line (0-16)
        list(range(0, 17)),
        # Right eyebrow (17-21)
        list(range(17, 22)),
        # Left eyebrow (22-26)
        list(range(22, 27)),
        # Nose bridge (27-30)
        list(range(27, 31)),
        # Nose base (31-35)
        list(range(31, 36)),
        # Right eye (36-41)
        list(range(36, 42)) + [36],
        # Left eye (42-47)
        list(range(42, 48)) + [42],
        # Outer lip (48-59)
        list(range(48, 60)) + [48],
        # Inner lip (60-67)
        list(range(60, 68)) + [60],
    ]
    
    # Draw connections
    for connection in connections:
        for i in range(len(connection) - 1):
            pt1 = tuple(landmarks[connection[i]])
            pt2 = tuple(landmarks[connection[i + 1]])
            cv2.line(image, pt1, pt2, color, 1)
    
    # Draw landmark points
    for i, (x, y) in enumerate(landmarks):
        # Different colors for different facial regions
        if i < 17:  # Jaw
            pt_color = (0, 255, 0)  # Green
        elif i < 27:  # Eyebrows
            pt_color = (255, 0, 0)  # Blue
        elif i < 36:  # Nose
            pt_color = (0, 255, 255)  # Yellow
        elif i < 48:  # Eyes
            pt_color = (255, 255, 0)  # Cyan
        else:  # Mouth
            pt_color = (0, 0, 255)  # Red
        
        cv2.circle(image, (x, y), radius, pt_color, -1)
        cv2.circle(image, (x, y), radius + 1, (255, 255, 255), 1)
    
    return image


def detect_face_opencv(frame):
    """Detect face using OpenCV Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        # Return the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        return faces[0]
    return None


def main():
    # Configuration
    MODEL_PATH = 'checkpoints_landmarks/checkpoint_epoch_190.pth'
    IMG_SIZE = 112
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("Real-time Facial Landmark Detection")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print("="*60)
    
    # Load model
    model = load_model(MODEL_PATH, DEVICE)
    transform = get_transform()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\nCamera opened successfully!")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        # Detect face
        face_bbox = detect_face_opencv(frame)
        
        if face_bbox is not None:
            x, y, w, h = face_bbox
            
            # Add padding
            padding = 0.2
            x_pad = int(w * padding)
            y_pad = int(h * padding)
            x1 = max(0, x - x_pad)
            y1 = max(0, y - y_pad)
            x2 = min(frame_width, x + w + x_pad)
            y2 = min(frame_height, y + h + y_pad)
            
            # Draw face bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Crop face region
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                # Prepare face for model
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_resized = face_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                
                # Transform and add batch dimension
                face_tensor = transform(face_resized).unsqueeze(0).to(DEVICE)
                
                # Predict landmarks
                with torch.no_grad():
                    _, pred_landmarks, heatmaps = model(face_tensor)
                
                # Convert landmarks to pixel coordinates
                h_map, w_map = heatmaps.size(2), heatmaps.size(3)
                landmarks = pred_landmarks[0].cpu().numpy()
                landmarks[:, 0] = landmarks[:, 0] / (w_map - 1)
                landmarks[:, 1] = landmarks[:, 1] / (h_map - 1)
                
                # Scale landmarks to face crop size
                crop_w = x2 - x1
                crop_h = y2 - y1
                landmarks[:, 0] *= crop_w
                landmarks[:, 1] *= crop_h
                
                # Translate landmarks to frame coordinates
                landmarks[:, 0] += x1
                landmarks[:, 1] += y1
                
                # Draw landmarks on frame
                frame = draw_landmarks(frame, landmarks)
                
                # Display info
                cv2.putText(frame, f"Landmarks: 68 points detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
        
        # Display FPS
        frame_count += 1
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Facial Landmark Detection (Press Q to quit)', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'landmark_capture_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed. Goodbye!")


if __name__ == '__main__':
    main()