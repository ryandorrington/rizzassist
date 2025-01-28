import os
from PIL import Image


def split_image(input_path, output_dir='split_images'):
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
    
    print(f"Split images saved in {output_dir}/")

def process_profile_directories(profiles_dir='profiles'):
    # Iterate through each subdirectory in profiles/
    for subdir in os.listdir(profiles_dir):
        subdir_path = os.path.join(profiles_dir, subdir)
        
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
            
        # Path to screenshots directory
        screenshots_dir = os.path.join(subdir_path, 'screenshots')
        
        # Skip if screenshots directory doesn't exist
        if not os.path.exists(screenshots_dir):
            continue
            
        # Create split_images directory within the subdirectory
        split_images_dir = os.path.join(subdir_path, 'split_images')
        
        # Process each PNG file in screenshots directory
        for filename in os.listdir(screenshots_dir):
            if filename.lower().endswith('.png'):
                input_path = os.path.join(screenshots_dir, filename)
                split_image(input_path, split_images_dir)

if __name__ == "__main__":
    process_profile_directories()
