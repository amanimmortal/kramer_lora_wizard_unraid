import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_tags(image_path: str, categories: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """Get top k tags for an image using CLIP model"""
    try:
        image = Image.open(image_path)
        inputs = processor(
            text=categories,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Get image and text features
        image_features = model.get_image_features(**{k:v for k,v in inputs.items() if k != 'input_ids'})
        text_features = model.get_text_features(inputs.input_ids)
        
        # Calculate similarity scores
        similarity = torch.nn.functional.cosine_similarity(
            image_features, text_features
        )
        
        # Get top k scores and indices
        top_scores, top_indices = similarity.topk(top_k)
        
        # Return tags and scores
        return [(categories[i], score.item()) for i, score in zip(top_indices, top_scores)]
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return []

def get_default_categories() -> List[str]:
    """Get default categories for image classification"""
    return [
        "landscape", "portrait", "abstract", "nature", "urban",
        "indoor", "outdoor", "day", "night", "colorful",
        "black and white", "macro", "aerial", "underwater", "street",
        "architecture", "wildlife", "food", "people", "vehicle"
    ] 