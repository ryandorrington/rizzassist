import os
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class FaceCropper:
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close-range, 1 for far-range
            min_detection_confidence=0.5
        )

    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the primary face in the image and return its bounding box.
        Returns (x_center, y_center, width, height) or None if no face found.
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)

        if not results.detections:
            return None

        # Get the first (most confident) face detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # Convert relative coordinates to absolute
        h, w = image.shape[:2]
        x_center = int((bbox.xmin + bbox.width/2) * w)
        y_center = int((bbox.ymin + bbox.height/2) * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        return (x_center, y_center, width, height)

    def crop_around_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face and crop a square region around it.
        Returns None if no face is detected.
        """
        face_bbox = self.detect_face(image)
        if face_bbox is None:
            return None

        x_center, y_center, face_w, face_h = face_bbox
        
        # Make crop region square and larger than the face
        crop_size = int(max(face_w, face_h) * 2)  # Change 1.8 to desired multiplier
        
        # Calculate crop boundaries
        h, w = image.shape[:2]
        x1 = max(0, x_center - crop_size//2)
        y1 = max(0, y_center - crop_size//2)
        x2 = min(w, x_center + crop_size//2)
        y2 = min(h, y_center + crop_size//2)
        
        # Crop and resize
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:  # Check if crop is empty
            return None
            
        return cv2.resize(cropped, (self.target_size, self.target_size))

def process_directory(base_dir: str):
    """Process all images in directory and save face-centered crops."""
    cropper = FaceCropper(target_size=224)
    
    # Create output directory
    output_dir = f'cropped_{os.path.basename(base_dir)}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all image files
    for profile_dir in os.listdir(base_dir):
        profile_path = os.path.join(base_dir, profile_dir)
        pictures_dir = os.path.join(profile_path, 'pictures')
        
        if not os.path.isdir(pictures_dir):
            continue
            
        for file in os.listdir(pictures_dir):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            input_path = os.path.join(pictures_dir, file)
            output_path = os.path.join(output_dir, f"{profile_dir}_{file}")
            
            # Process image
            try:
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Failed to load image: {input_path}")
                    continue
                    
                cropped = cropper.crop_around_face(image)
                if cropped is not None:
                    cv2.imwrite(output_path, cropped)
                else:
                    print(f"No face detected in: {input_path}")
                    
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")

if __name__ == "__main__":
    # Process both Tinder and Bumble profiles
    for app_dir in ['tinder_profiles', 'bumble_profiles']:
        if os.path.exists(app_dir):
            print(f"\nProcessing {app_dir}...")
            process_directory(app_dir)
            print(f"Finished processing {app_dir}")
