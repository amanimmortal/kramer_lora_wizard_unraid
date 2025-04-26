import os
import json
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from app.services.image_service import get_project_image_path, ensure_project_dirs
from app.services.model_service import get_image_tags, get_default_categories

class ProjectService:
    def __init__(self, base_path: str = "data/datasets"):
        self.base_path = Path(base_path)
        
    def create_project(self) -> str:
        """Create a new project and return its ID"""
        project_id = str(uuid.uuid4())
        ensure_project_dirs(project_id)
        return project_id
        
    def get_project_metadata(self, project_id: str) -> Dict:
        """Get project metadata"""
        metadata_path = self.base_path / project_id / "metadata" / "project.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}
        
    def save_project_metadata(self, project_id: str, metadata: Dict) -> None:
        """Save project metadata"""
        metadata_path = self.base_path / project_id / "metadata" / "project.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
    def process_images(self, project_id: str, categories: Optional[List[str]] = None) -> Dict:
        """Process all images in project directory and generate tags"""
        if categories is None:
            categories = get_default_categories()
            
        image_dir = get_project_image_path(project_id)
        results = {}
        
        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_file)
                tags = get_image_tags(img_path, categories)
                results[img_file] = {
                    'tags': tags,
                    'processed_at': datetime.now().isoformat()
                }
                
        # Save results
        output_path = self.base_path / project_id / "output" / "image_tags.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        return results 