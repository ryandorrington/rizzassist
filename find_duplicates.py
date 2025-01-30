import os
from PIL import Image # type: ignore
import imagehash
from collections import defaultdict
from typing import Union
from pathlib import Path

def find_duplicates(split_images_dir: Union[str, Path], auto_remove: bool = False) -> None:
    """Find and optionally remove duplicate images in a directory.
    
    Args:
        split_images_dir: Directory containing images to check for duplicates
        auto_remove: If True, removes duplicate images keeping first occurrence
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        OSError: If there are permission issues
    """
    # Dictionary to store hash -> [file_paths]
    hash_dict = defaultdict(list)
    
    if not os.path.exists(split_images_dir):
        raise FileNotFoundError(f"Directory not found: {split_images_dir}")
            
    # Process each image in the directory
    for filename in os.listdir(split_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(split_images_dir, filename)
            try:
                # Calculate image hash
                with Image.open(image_path) as img:
                    hash = str(imagehash.average_hash(img))
                    hash_dict[hash].append(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    # Print duplicate groups and optionally remove duplicates
    for hash, file_list in hash_dict.items():
        if len(file_list) > 1:
            print(f"\nFound {len(file_list)} duplicates:")
            for file_path in file_list:
                print(f"  - {file_path}")
            
            if auto_remove:
                # Keep the first file, remove the rest
                for file_path in file_list[1:]:
                    try:
                        os.remove(file_path)
                        print(f"Removed duplicate: {file_path}")
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
