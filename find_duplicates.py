import os
from PIL import Image
import imagehash
from collections import defaultdict

def find_duplicate_images(profiles_dir='profiles', auto_remove=False):
    """
    Find duplicate images in all 'pictures' directories by comparing their content.
    Uses perceptual hashing to identify visually identical images.
    
    Args:
        profiles_dir (str): Directory containing profile folders
        auto_remove (bool): If True, automatically remove duplicate images
    """
    # Dictionary to store hash -> [file_paths]
    hash_dict = defaultdict(list)
    
    # Scan all pictures directories
    for subdir in os.listdir(profiles_dir):
        subdir_path = os.path.join(profiles_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        pictures_dir = os.path.join(subdir_path, 'pictures')
        if not os.path.exists(pictures_dir):
            continue
            
        # Process each image in the pictures directory
        for filename in os.listdir(pictures_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(pictures_dir, filename)
                try:
                    # Calculate image hash
                    with Image.open(image_path) as img:
                        hash = str(imagehash.average_hash(img))
                        hash_dict[hash].append(image_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    # Print duplicate groups and optionally remove duplicates
    print("\nFound duplicate images:")
    found_duplicates = False
    for hash, file_list in hash_dict.items():
        if len(file_list) > 1:
            found_duplicates = True
            print("\nDuplicate group:")
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
    
    if not found_duplicates:
        print("No duplicates found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Find and optionally remove duplicate images.')
    parser.add_argument('--remove', action='store_true', help='Automatically remove duplicate images')
    args = parser.parse_args()
    
    find_duplicate_images(auto_remove=args.remove) 
