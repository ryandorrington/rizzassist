import os
from PIL import Image
from typing import Union

def stitch_images(subdir_path: str, images_directory: str, single_width: int, single_height: int, max_images: int = 6) -> None:
    """
    Stitch together images from a profile's pictures directory in a grid layout.
    
    Args:
        subdir_path: Path to the profile directory containing 'pictures' folder
        images_directory: Path to the directory containing images to stitch
        single_width: Width of each individual image
        single_height: Height of each individual image
        max_images: Maximum number of images to include (6 or 9)
        
    Raises:
        ValueError: If max_images is not 6 or 9
        FileNotFoundError: If pictures directory doesn't exist
        PIL.Image.Error: If there are issues processing images
    """
    if max_images not in (6, 9):
        raise ValueError("max_images must be either 6 or 9")
        
    # Calculate grid dimensions based on max_images
    cols = 3
    rows = max_images // 3
    
    # Size for the final stitched image
    grid_width = single_width * cols
    grid_height = single_height * rows
    
    if not os.path.exists(images_directory):
        raise FileNotFoundError(f"Images directory not found: {images_directory}")
    
    # Get all PNG files in the pictures directory
    image_files = sorted([f for f in os.listdir(images_directory) if f.lower().endswith('.png')])
    if not image_files:
        return
    
    # Create new blank image with white background
    stitched = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Place each image in the grid
    for idx, image_file in enumerate(image_files[:max_images]):
        # Open the image
        img_path = os.path.join(images_directory, image_file)
        with Image.open(img_path) as img:
            # Calculate position in grid
            row = idx // 3  # 0 for first row, 1 for second row
            col = idx % 3   # 0, 1, or 2 for column position
            
            # Calculate paste coordinates
            x = col * single_width
            y = row * single_height
            
            # Paste the image
            stitched.paste(img, (x, y))
    
    # Save the stitched image
    output_path = os.path.join(subdir_path, 'stitched_profile.png')
    stitched.save(output_path)
