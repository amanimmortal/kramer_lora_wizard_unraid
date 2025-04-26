import os
import shutil
from pathlib import Path
from typing import List
from PIL import Image

def get_project_image_path(project_id: str) -> str:
    """Get the image directory path for a project"""
    return os.path.join("data/datasets", project_id, "images")

def ensure_project_dirs(project_id: str) -> None:
    """Ensure all required project directories exist"""
    base_path = Path("data/datasets") / project_id
    dirs = ["images", "output", "metadata", "log", "reg"]
    for dir_name in dirs:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)

def save_images(project_id: str, image_paths: List[str]) -> List[str]:
    """Save images to project directory and return their new paths"""
    project_img_dir = get_project_image_path(project_id)
    saved_paths = []
    
    for img_path in image_paths:
        try:
            # Open and verify image
            img = Image.open(img_path)
            img.verify()
            
            # Get filename and create new path
            filename = os.path.basename(img_path)
            new_path = os.path.join(project_img_dir, filename)
            
            # Copy image to project directory
            shutil.copy2(img_path, new_path)
            saved_paths.append(new_path)
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            continue
            
    return saved_paths

def get_project_images(project_id: str) -> List[str]:
    """Get list of all images in project directory"""
    image_dir = get_project_image_path(project_id)
    if not os.path.exists(image_dir):
        return []
        
    return [
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ] 