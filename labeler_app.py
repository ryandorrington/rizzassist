import os
import csv
import glob
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from typing import List, Tuple

class ProfileLabelerApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Dating Profile Labeler")

        # Output CSV file
        self.output_csv = "profile_labels.csv"
        self._initialize_csv()
        
        # List of (stitched_image_path, pictures_dir) for all profiles
        self.profiles = self._collect_profiles()
        self.current_index = self._get_last_labeled_index()
        
        # Rating and match state
        self.current_rating = None
        
        # Create scrollable canvas for image
        self.canvas = tk.Canvas(self.master)
        self.scrollbar = tk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True, pady=10)
        
        # Create window in canvas for image
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Main UI elements
        self.image_label = tk.Label(self.scrollable_frame)
        self.image_label.pack(pady=10)
        
        # Create frame for progress info
        self.info_frame = tk.Frame(self.master)
        self.info_frame.pack(pady=5)
        
        self.progress_label = tk.Label(self.info_frame, text="", font=("Arial", 14))
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        self.profile_label = tk.Label(self.info_frame, text="", font=("Arial", 14))
        self.profile_label.pack(side=tk.LEFT, padx=10)
        
        # Handle key presses
        self.master.bind("<Key>", self._on_key_press)
        
        # Configure scrolling
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        ))
        
        # Bind mouse wheel to scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Load the first profile if available
        if self.profiles:
            self._load_profile(self.current_index)
        else:
            messagebox.showinfo("No Profiles", "No profiles found to label.")
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _collect_profiles(self) -> List[Tuple[str, str]]:
        """
        Collect all stitched image paths (e.g., 'stitched.png') along with the
        corresponding 'pictures' directory from both tinder_profiles and bumble_profiles.
        """
        profiles = []
        for base in ["bumble_profiles", "tinder_profiles"]:
            if not os.path.isdir(base):
                continue
            # List all subdirectories in base
            for subdir in os.listdir(base):
                subdir_path = os.path.join(base, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                # Look for a stitched image
                # E.g., might be named "stitched.png" or similar
                stitched_images = glob.glob(os.path.join(subdir_path, "*stitched*.*"))
                if not stitched_images:
                    # If no stitched image found, skip
                    continue
                # For this example, assume there's only one stitched image per profile
                stitched_path = stitched_images[0]
                
                # The pictures directory
                pictures_dir = os.path.join(subdir_path, "pictures")
                # Some profiles might be in "screenshots" for tinder, so we account for that
                if not os.path.isdir(pictures_dir):
                    # If no pictures subfolder, try "screenshots"
                    alt_dir = os.path.join(subdir_path, "screenshots")
                    if os.path.isdir(alt_dir):
                        pictures_dir = alt_dir
                
                profiles.append((stitched_path, pictures_dir))
        
        return profiles
    
    def _initialize_csv(self):
        """Create the CSV file with headers if it doesn't exist."""
        if not os.path.isfile(self.output_csv):
            with open(self.output_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "rating", "match"])
    
    def _load_profile(self, index: int):
        """Load and display the stitched image from the profile at self.profiles[index]."""
        if index < 0 or index >= len(self.profiles):
            return
        
        stitched_path, _ = self.profiles[index]
        try:
            pil_img = Image.open(stitched_path)
            # Use Resampling.LANCZOS instead of deprecated ANTIALIAS
            pil_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=self.tk_img)
        except Exception as e:
            messagebox.showwarning("Image Load Error", f"Could not load {stitched_path}\n{e}")
            self._next_profile()
            return
        
        # Reset rating
        self.current_rating = None
        
        # Get profile name from path
        profile_name = os.path.basename(os.path.dirname(stitched_path))
        
        # Update progress and profile labels
        self.progress_label.config(
            text=f"Profile {index+1}/{len(self.profiles)}"
        )
        self.profile_label.config(text=f"Profile: {profile_name}")
    
    def _on_key_press(self, event):
        """
        Handle numeric rating (1-5) and arrow keys for match(0) or match(1).
        After setting rating and match, automatically save CSV rows and go to next profile.
        """
        key = event.keysym
        
        if key in ["1", "2", "3", "4", "5"]:
            self.current_rating = int(key)
            # Visually indicate rating chosen, if desired
            self.progress_label.config(text=f"{self.progress_label.cget('text')} - Rating: {self.current_rating}")
        
        elif key == "Left" or key == "Right":
            # We need to have a rating chosen first
            if self.current_rating is None:
                messagebox.showinfo("No Rating", "Please select a rating (1-5) first.")
                return
            
            match_value = 1 if key == "Right" else 0
            # Save data for each original photo in pictures_dir
            self._save_label(match_value)
            # Move to next profile
            self._next_profile()
    
    def _save_label(self, match_value: int):
        """
        Save rows in the CSV for each image found in the current profile's pictures directory.
        Columns: image_name, rating, match
        """
        stitched_path, pictures_dir = self.profiles[self.current_index]
        if not os.path.isdir(pictures_dir):
            return
        
        profile_name = os.path.basename(os.path.dirname(stitched_path))
        photo_paths = glob.glob(os.path.join(pictures_dir, "*.*"))
        
        with open(self.output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for photo_path in photo_paths:
                photo_name = os.path.basename(photo_path)
                photo_name_with_profile = f"{profile_name}_{photo_name}"
                writer.writerow([photo_name_with_profile, self.current_rating, match_value])
    
    def _next_profile(self):
        """Advance to the next profile in the list."""
        self.current_index += 1
        if self.current_index >= len(self.profiles):
            messagebox.showinfo("Done", "All profiles have been labeled!")
            self.master.quit()
        else:
            self._load_profile(self.current_index)

    def _get_last_labeled_index(self) -> int:
        """
        Read the CSV file to determine the last labeled profile and return the next index.
        """
        if not os.path.exists(self.output_csv):
            return 0
            
        labeled_photos = set()
        with open(self.output_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                photo_path = row[0]
                # Extract original photo name by removing profile prefix
                original_photo = '_'.join(photo_path.split('_')[2:]) if '_' in photo_path else photo_path
                # Find which profile this photo belongs to
                for i, (_, pictures_dir) in enumerate(self.profiles):
                    if os.path.exists(os.path.join(pictures_dir, original_photo)):
                        labeled_photos.add(i)
                        break
        
        # If we found any labeled profiles, return the next unlabeled index
        if labeled_photos:
            next_index = max(labeled_photos) + 1
            return min(next_index, len(self.profiles))
        return 0

def main():
    root = tk.Tk()
    app = ProfileLabelerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
