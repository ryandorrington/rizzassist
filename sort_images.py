import os
from PIL import Image
import shutil

def get_pixel_color(image_path, x, y):
    """
    Get RGB value of a pixel at coordinates (x,y) in an image
    
    Args:
        image_path (str): Path to the image file
        x (int): X coordinate of the pixel
        y (int): Y coordinate of the pixel
    
    Returns:
        tuple: RGB values as (r,g,b)
    """
    # Open the image
    with Image.open(image_path) as img:
        # Convert image to RGB mode if it isn't already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get the RGB value at the specified pixel
        r, g, b = img.getpixel((x, y))
        return (r, g, b)

def sort_images(split_images_dir: str, bio_info_dir: str, pictures_dir: str) -> None:
    """Sort images into bio_info and pictures directories based on pixel color.
    
    Args:
        split_images_dir: Directory containing split images
        bio_info_dir: Directory to store bio info images
        pictures_dir: Directory to store profile pictures
    """

    # Process each PNG file in split_images directory
    for filename in os.listdir(split_images_dir):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(split_images_dir, filename)
            
            # Get pixel color at (440, 117)
            r, g, b = get_pixel_color(image_path, 440, 10)
            
            # Determine destination based on RGB value
            if abs(r - 254) <= 5 and abs(g - 248) <= 5 and abs(b - 219) <= 5:
                dest_dir = bio_info_dir
                print(f"Bio info image: {filename} ({r}, {g}, {b})")
            else:
                dest_dir = pictures_dir
                print(f"Picture image: {filename} ({r}, {g}, {b})")
                
            # Copy image to appropriate directory
            shutil.copy2(image_path, os.path.join(dest_dir, filename))
