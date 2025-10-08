"""
Phase 1A: Generate 68-point facial landmarks for RAF-DB dataset
Using face-alignment library as PFA substitute
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import face_alignment
import torch

def generate_landmarks_for_rafdb(dataset_root, output_dir='./landmarks_rafdb'):
    """
    Generate 68-point facial landmarks for all RAF-DB images
    
    Args:
        dataset_root: Path to RAF-DB dataset (e.g., './archive/DATASET')
        output_dir: Where to save landmark files
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("RAF-DB Landmark Generation (Phase 1A)")
    print("="*70)
    print(f"Dataset: {dataset_root}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Initialize face alignment model (acts as PFA substitute)
    print("\nInitializing face alignment model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        device=device,
        flip_input=False
    )
    print("✅ Model loaded!")
    
    # Process both train and test sets
    total_processed = 0
    total_failed = 0
    
    for split in ['train', 'test']:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} set")
        print(f"{'='*70}")
        
        split_dir = os.path.join(dataset_root, split)
        split_output_dir = os.path.join(output_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"❌ {split} folder not found, skipping...")
            continue
        
        # Process each emotion folder
        for emotion_id in range(1, 8):
            emotion_dir = os.path.join(split_dir, str(emotion_id))
            emotion_output_dir = os.path.join(split_output_dir, str(emotion_id))
            
            if not os.path.exists(emotion_dir):
                continue
            
            # Create output folder
            os.makedirs(emotion_output_dir, exist_ok=True)
            
            # Get all images
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"\nEmotion {emotion_id}: Processing {len(image_files)} images...")
            
            failed_count = 0
            
            # Process each image
            for img_name in tqdm(image_files, desc=f"Emotion {emotion_id}"):
                img_path = os.path.join(emotion_dir, img_name)
                landmark_name = img_name.rsplit('.', 1)[0] + '.npy'
                landmark_path = os.path.join(emotion_output_dir, landmark_name)
                
                # Skip if already processed
                if os.path.exists(landmark_path):
                    total_processed += 1
                    continue
                
                try:
                    # Load image
                    image = Image.open(img_path).convert('RGB')
                    image_np = np.array(image)
                    
                    # Detect landmarks
                    landmarks = fa.get_landmarks(image_np)
                    
                    if landmarks is not None and len(landmarks) > 0:
                        # Take first face (should only be one in RAF-DB)
                        face_landmarks = landmarks[0]  # Shape: (68, 2)
                        
                        # Normalize to [0, 1]
                        h, w = image_np.shape[:2]
                        face_landmarks_norm = face_landmarks.copy()
                        face_landmarks_norm[:, 0] /= w
                        face_landmarks_norm[:, 1] /= h
                        
                        # Save landmarks
                        np.save(landmark_path, face_landmarks_norm.astype(np.float32))
                        total_processed += 1
                    else:
                        # No face detected - use template landmarks
                        print(f"\n⚠️  No face detected in {img_name}, using template")
                        template_landmarks = get_template_landmarks()
                        np.save(landmark_path, template_landmarks)
                        failed_count += 1
                        total_failed += 1
                        
                except Exception as e:
                    print(f"\n❌ Error processing {img_name}: {e}")
                    # Save template landmarks as fallback
                    template_landmarks = get_template_landmarks()
                    np.save(landmark_path, template_landmarks)
                    failed_count += 1
                    total_failed += 1
            
            if failed_count > 0:
                print(f"   ⚠️  {failed_count} images failed, used template landmarks")
    
    print("\n" + "="*70)
    print("LANDMARK GENERATION COMPLETE")
    print("="*70)
    print(f"Total processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    print(f"Landmarks saved to: {output_dir}")
    print("="*70)


def get_template_landmarks():
    """Fallback template landmarks for failed detections"""
    landmarks = np.array([
        # Jaw line (0-16)
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
        # Outer mouth (48-59)
        [0.38, 0.72], [0.41, 0.71], [0.44, 0.70], [0.47, 0.70], [0.50, 0.70],
        [0.53, 0.70], [0.56, 0.70], [0.59, 0.71], [0.62, 0.72],
        [0.59, 0.76], [0.56, 0.77], [0.53, 0.77], [0.50, 0.77],
        [0.47, 0.77], [0.44, 0.77], [0.41, 0.76],
        # Inner mouth (60-67)
        [0.41, 0.73], [0.44, 0.72], [0.47, 0.72], [0.50, 0.72],
        [0.53, 0.72], [0.56, 0.72], [0.59, 0.73], [0.50, 0.74],
    ], dtype=np.float32)
    return landmarks


if __name__ == '__main__':
    # Configuration
    DATASET_ROOT = './archive/DATASET'
    OUTPUT_DIR = './landmarks_rafdb'
    
    print("\nIMPORTANT: This script requires face-alignment library")
    print("Install with: pip install face-alignment")
    print("\nThis will take ~10-15 minutes for the full RAF-DB dataset.")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        generate_landmarks_for_rafdb(DATASET_ROOT, OUTPUT_DIR)
        
        print("\n✅ SUCCESS! Landmarks generated.")
        print("\nNext step: Run the landmark training script (Phase 1B)")
        print("The training script will now use these real landmarks instead of templates.")
        
    except ImportError:
        print("\n❌ ERROR: face-alignment library not found!")
        print("\nPlease install it:")
        print("  pip install face-alignment")
        print("\nThen run this script again.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()