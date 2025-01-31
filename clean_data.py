import os
import shutil
from PIL import Image # type: ignore
from find_duplicates import find_duplicates
from split_image import split_image
from sort_images import sort_images
from stitch_images import stitch_images

def clean_tinder_data() -> None:
    """Process and organize Tinder profile data."""
    for subdir in os.listdir('tinder_profiles'):
        subdir_path = os.path.join('tinder_profiles', subdir)
        
        print(subdir_path)
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
            
        # Path to screenshots directory
        screenshots_dir = os.path.join(subdir_path, 'screenshots')
        
        # Skip if screenshots directory doesn't exist
        if not os.path.exists(screenshots_dir):
            continue
        
        find_duplicates(screenshots_dir, auto_remove=True)

        # Only stitch images if stitched_profile.png doesn't exist
        stitched_image_path = os.path.join(subdir_path, 'stitched_profile.png')
        if not os.path.exists(stitched_image_path):
            try:
                stitch_images(subdir_path, screenshots_dir, single_width=600, single_height=1110, max_images=9)
            except Exception as e:
                print(f"Error stitching images for {subdir}: {e}")

def clean_bumble_data() -> None:
    """Process and organize Bumble profile data.
    
    Processes each profile directory in bumble_profiles:
    1. Splits screenshots into left/right images
    2. Sorts images into bio_info and pictures directories
    3. Creates a stitched profile image
    """
    # Iterate through each subdirectory
    for subdir in os.listdir('bumble_profiles'):
        subdir_path = os.path.join('bumble_profiles', subdir)
        
        print(subdir_path)
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
            
        # Path to screenshots directory
        screenshots_dir = os.path.join(subdir_path, 'screenshots')
        
        # Skip if screenshots directory doesn't exist
        if not os.path.exists(screenshots_dir):
            continue
            
        # Create directories within the subdirectory
        split_images_dir = os.path.join(subdir_path, 'split_images')
        pictures_dir = os.path.join(subdir_path, 'pictures')
        bio_info_dir = os.path.join(subdir_path, 'bio_info')
        
        # Only split images if split_images directory is empty
        if not os.path.exists(split_images_dir) or not os.listdir(split_images_dir):
            os.makedirs(split_images_dir, exist_ok=True)
            # Process each PNG file in screenshots directory
            for filename in os.listdir(screenshots_dir):
                if filename.lower().endswith('.png'):
                    input_path = os.path.join(screenshots_dir, filename)
                    split_image(input_path, split_images_dir)
            
            # Check for and remove duplicates
            find_duplicates(split_images_dir, auto_remove=True)

        if not os.path.exists(pictures_dir) or not os.listdir(pictures_dir):
            os.makedirs(pictures_dir, exist_ok=True)
            os.makedirs(bio_info_dir, exist_ok=True)
            sort_images(split_images_dir, bio_info_dir, pictures_dir)

        # Only stitch images if stitched_profile.png doesn't exist
        stitched_image_path = os.path.join(subdir_path, 'stitched_profile.png')
        if not os.path.exists(stitched_image_path):
            try:
                stitch_images(subdir_path, pictures_dir, single_width=880, single_height=1170, max_images=6)
            except Exception as e:
                print(f"Error stitching images for {subdir}: {e}")

def main() -> None:
    """Main entry point for the script.
    
    Gets APP environment variable and calls appropriate cleaning function.
    Raises ValueError if APP is not 'tinder' or 'bumble'.
    """
    # Get app type from environment variable
    app = os.getenv('APP')

    if app == 'tinder':
        clean_tinder_data()
    elif app == 'bumble':
        clean_bumble_data()
    else:
        raise ValueError("APP environment variable must be set to 'tinder' or 'bumble'")

if __name__ == "__main__":
    main()
