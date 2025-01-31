import os
from pathlib import Path
from typing import Dict

def count_profile_images(base_dir: str, target_dir: str) -> Dict[str, int]:
    """Count images in each profile's target directory.
    
    Args:
        base_dir: Base directory (bumble_profiles or tinder_profiles)
        target_dir: Target directory to count images in (pictures or screenshots)
        
    Returns:
        Dictionary mapping profile names to image counts
    """
    counts = {}
    total = 0
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return counts
        
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        images_dir = os.path.join(subdir_path, target_dir)
        if not os.path.exists(images_dir):
            counts[subdir] = 0
            continue
            
        image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
        counts[subdir] = image_count
        total += image_count
    
    return counts, total

def main() -> None:
    """Count and display image statistics for both apps."""
    # Count Bumble images
    print("Bumble Profiles:")
    bumble_counts, bumble_total = count_profile_images('bumble_profiles', 'pictures')
    for profile, count in bumble_counts.items():
        print(f"  {profile}: {count} images")
    print(f"\nTotal Bumble images: {bumble_total}")
    
    # Count Tinder images
    print("\nTinder Profiles:")
    tinder_counts, tinder_total = count_profile_images('tinder_profiles', 'screenshots')
    for profile, count in tinder_counts.items():
        print(f"  {profile}: {count} images")
    print(f"\nTotal Tinder images: {tinder_total}")
    
    print(f"\nTotal images across both apps: {bumble_total + tinder_total}")

if __name__ == "__main__":
    main() 
