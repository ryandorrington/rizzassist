import os
from PIL import Image # type: ignore
from pathlib import Path
from typing import Union


def split_image(input_path: Union[str, Path], output_dir: Union[str, Path] = 'split_images') -> None:
    """Split an image horizontally into two equal parts.
    
    Args:
        input_path: Path to the input image file
        output_dir: Directory to save the split images
        
    Raises:
        FileNotFoundError: If input_path doesn't exist
        PIL.Image.Error: If there are issues processing the image
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the image
    img = Image.open(input_path)
    
    # Get the width and height
    width, height = img.size
    
    # Split the image into two halves
    left_half = img.crop((0, 0, width//2, height))
    right_half = img.crop((width//2, 0, width, height))
    
    # Generate output filenames based on input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Save the halves
    left_half.save(os.path.join(output_dir, f"{base_name}_left.png"))
    right_half.save(os.path.join(output_dir, f"{base_name}_right.png"))
    
    print(f"Split images saved in {output_dir}/{base_name}")
