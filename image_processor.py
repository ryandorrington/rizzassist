import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def preprocess_image(img_array):
    """
    Preprocess a single image that's already 224x224
    Returns normalized image in correct format for the model
    """
    # Convert to float32 and ensure only 3 channels (RGB)
    img = img_array.astype(np.float32)[:, :, :3]
    
    # Rearrange axes from (H,W,C) to (C,H,W)
    img = np.moveaxis(img, [2,0,1], [0,1,2])
    
    # Normalize
    img /= 255.0
    img -= 0.5
    img /= 0.5
    
    return img

def process_labeled_images(input_dir: str, output_dir: str, labels_csv: str = "profile_labels.csv", batch_size: int = 32):
    """
    Process only the images listed in labels_csv and save them in batches.
    """
    # Load and validate labels file
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    
    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} labels from {labels_csv}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize batch tracking
    current_batch = []
    current_batch_labels = []
    current_batch_files = []
    batch_count = 0
    processed = 0
    skipped = 0
    
    # Process each labeled image
    print("Processing labeled images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(input_dir, row['image_name'])
        
        try:
            # Verify image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                skipped += 1
                continue
            
            # Load and verify image
            img = Image.open(image_path)
            if img.size != (224, 224):
                print(f"Warning: Skipping {image_path}. Image is not 224x224")
                skipped += 1
                continue
            
            # Process image
            img_array = np.array(img)
            processed_img = preprocess_image(img_array)
            
            # Add to current batch
            current_batch.append(processed_img)
            current_batch_labels.append((row['rating'], row['match']))
            current_batch_files.append(row['image_name'])
            processed += 1
            
            # Save batch if it's full or last item
            if len(current_batch) == batch_size or idx == len(df) - 1:
                if current_batch:  # Only save if we have images
                    # Create arrays
                    batch_array = np.stack(current_batch)
                    labels_array = np.array(current_batch_labels)
                    
                    # Save batch data
                    batch_file = os.path.join(output_dir, f'batch_{batch_count:04d}.npy')
                    labels_file = os.path.join(output_dir, f'labels_{batch_count:04d}.npy')
                    files_file = os.path.join(output_dir, f'files_{batch_count:04d}.txt')
                    
                    np.save(batch_file, batch_array)
                    np.save(labels_file, labels_array)
                    with open(files_file, 'w') as f:
                        f.write('\n'.join(current_batch_files))
                    
                    print(f"\nBatch {batch_count}:")
                    print(f"  Images shape: {batch_array.shape}")
                    print(f"  Labels shape: {labels_array.shape}")
                    print(f"  Files saved: {len(current_batch_files)}")
                    
                    # Reset batch
                    current_batch = []
                    current_batch_labels = []
                    current_batch_files = []
                    batch_count += 1
                    
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            skipped += 1
    
    print(f"\nProcessing complete:")
    print(f"  Processed: {processed} images")
    print(f"  Skipped: {skipped} images")
    print(f"  Total batches: {batch_count}")
    
    return batch_count > 0

def load_batch(output_dir: str, batch_num: int):
    """Load a batch of processed images and their labels"""
    batch_file = os.path.join(output_dir, f'batch_{batch_num:04d}.npy')
    labels_file = os.path.join(output_dir, f'labels_{batch_num:04d}.npy')
    files_file = os.path.join(output_dir, f'files_{batch_num:04d}.txt')
    
    images = np.load(batch_file)
    labels = np.load(labels_file)
    with open(files_file) as f:
        files = f.read().splitlines()
    
    return images, labels, files

if __name__ == "__main__":
    BATCH_SIZE = 32
    
    # Process both Tinder and Bumble datasets
    for app in ['tinder', 'bumble']:
        print(f"\nProcessing {app} profiles...")
        INPUT_DIR = f"cropped_{app}_profiles"
        OUTPUT_DIR = "processed_data"
        
        # Process dataset
        if process_labeled_images(INPUT_DIR, OUTPUT_DIR, batch_size=BATCH_SIZE):
            # Load first batch as example
            images, labels, files = load_batch(OUTPUT_DIR, 0)
            print(f"\nFirst {app} batch loaded:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Number of files: {len(files)}")
