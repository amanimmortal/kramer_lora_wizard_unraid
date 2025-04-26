from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import timm
import os
import logging
import json
from typing import List, Dict, Any
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import csv

class TaggerService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize BLIP
        self.logger.info("Initializing BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Initialize WD Tagger using timm
        self.logger.info("Initializing WD tagger model...")
        self.model_id = "SmilingWolf/wd-vit-tagger-v3"
        
        # Create ViT model with correct architecture
        self.wd_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=10861,
            img_size=448,
            class_token=False,
            global_pool='avg',
            fc_norm=False
        )
        
        # Check for model in persistent storage first
        persistent_model_path = "/app/data/models/wd14/model.safetensors"
        if os.path.exists(persistent_model_path):
            self.logger.info("Loading WD tagger weights from persistent storage...")
            weights_path = persistent_model_path
        else:
            # Load weights from Hugging Face
            self.logger.info("Loading WD tagger weights from Hugging Face...")
            weights_path = hf_hub_download(self.model_id, "model.safetensors")
            
            # Save to persistent storage if possible
            os.makedirs("/app/data/models/wd14", exist_ok=True)
            try:
                import shutil
                shutil.copy2(weights_path, persistent_model_path)
                self.logger.info("Saved model weights to persistent storage")
            except Exception as e:
                self.logger.warning(f"Could not save model to persistent storage: {e}")
        
        state_dict = load_file(weights_path)
        self.wd_model.load_state_dict(state_dict)
        self.wd_model.eval()
        
        # Load tag names similarly
        persistent_tags_path = "/app/data/models/wd14/selected_tags.csv"
        if os.path.exists(persistent_tags_path):
            self.logger.info("Loading tags from persistent storage...")
            self.tags = self._load_tags(persistent_tags_path)
        else:
            self.logger.info("Loading tags from Hugging Face...")
            self.tags = self._load_tags()
            
            # Save tags to persistent storage
            try:
                tags_path = hf_hub_download(self.model_id, "selected_tags.csv")
                shutil.copy2(tags_path, persistent_tags_path)
                self.logger.info("Saved tags to persistent storage")
            except Exception as e:
                self.logger.warning(f"Could not save tags to persistent storage: {e}")
        
        self.logger.info("Models initialized successfully")
        
    def _load_tags(self, tags_path: str = None) -> List[str]:
        """Load tag names from file path or model's repository"""
        try:
            if not tags_path:
                tags_path = hf_hub_download(self.model_id, "selected_tags.csv")
            
            tags = []
            with open(tags_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row and 'name' in row:  # Skip empty rows and ensure 'name' column exists
                        tag_name = row['name'].strip()
                        if tag_name:  # Skip empty tag names
                            tags.append(tag_name)
            return tags
        except Exception as e:
            self.logger.error(f"Error loading tags: {str(e)}")
            return []

    def get_blip_caption(self, image_path: str) -> str:
        """Generate a caption for an image using BLIP"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            self.logger.error(f"Error generating caption for {image_path}: {str(e)}")
            return ""

    def get_wd14_tags(self, image_path: str, general_threshold: float = 0.35, character_threshold: float = 0.35, max_tags: int = 50) -> list[str]:
        """Get Danbooru-style tags using WD tagger model"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((448, 448))  # Model expects 448x448
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
            # Normalize with model's stats
            mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
            image = (image - mean) / std
            
            image = image.unsqueeze(0)  # Add batch dimension
            
            # Get predictions
            with torch.no_grad():
                outputs = self.wd_model(image)
                probs = torch.sigmoid(outputs)[0]
                
                # Filter and sort tags based on confidence
                tags_probs = [(self.tags[i], float(probs[i])) for i in range(len(probs))]
                tags_probs.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
                
                # Separate general and character tags
                general_tags = []
                character_tags = []
                
                for tag, prob in tags_probs:
                    if prob < general_threshold:
                        continue
                        
                    # Character tags typically contain underscores and parentheses
                    if "_(" in tag and prob >= character_threshold:
                        character_tags.append(tag)
                    elif prob >= general_threshold:
                        general_tags.append(tag)
                
                # Combine and limit tags
                final_tags = character_tags + general_tags
                if max_tags > 0:
                    final_tags = final_tags[:max_tags]
                
                self.logger.debug(f"Generated {len(final_tags)} tags for {image_path}")
                return final_tags
                
        except Exception as e:
            self.logger.error(f"Error generating tags for {image_path}: {str(e)}")
            return []
        
    def auto_tag_image(self, image_path: str, mode: str = "wd14", general_threshold: float = 0.35, character_threshold: float = 0.35, max_tags: int = 50) -> list[str]:
        """Auto-tag an image using either WD14 tags or BLIP captioning
        
        Args:
            image_path: Path to the image file
            mode: Either "wd14" for Danbooru tags or "caption" for BLIP captions
            general_threshold: Confidence threshold for general tags
            character_threshold: Confidence threshold for character tags
            max_tags: Maximum number of tags to return
            
        Returns:
            List of tags or caption as list with single string
        """
        if mode == "wd14":
            return self.get_wd14_tags(image_path, general_threshold, character_threshold, max_tags)
        else:
            return [self.get_blip_caption(image_path)] 