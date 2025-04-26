from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List
from pydantic import BaseModel
from ..services.training import TrainingService
from ..models.lora import LoraProject, AutoTagSettings, LoraType
from app.services.tagger import TaggerService
import os
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from app.services.image_service import get_project_image_path
import logging
import json
import re
from fastapi.responses import PlainTextResponse

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(tags=["training"])
training_service = TrainingService()
tagger = TaggerService()

# --- Define Pydantic Model for Training Request --- 
class TrainingRequestPayload(BaseModel):
    baseModelName: str
    repeats: int
    settings: Dict[str, Any] # The loaded JSON template content

# --- Helper to get project data (similar to projects.py) ---
# NOTE: Consider moving this to a shared service/utils file
async def get_project_details(project_id: str) -> LoraProject:
    project_file_path = f"data/datasets/{project_id}/metadata/project.json"
    if not os.path.exists(project_file_path):
        logger.error(f"Project metadata file not found: {project_file_path}")
        raise HTTPException(status_code=404, detail="Project definition not found")
    try:
        with open(project_file_path, "r") as f:
            project_data = LoraProject.model_validate_json(f.read())
        return project_data
    except Exception as e:
        logger.error(f"Error reading project file {project_file_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not load project data")

# --- Modified Training Endpoint --- 
@router.post("/{project_id}/train")
async def start_training(project_id: str, payload: TrainingRequestPayload):
    """Prepares dataset structure and starts training a LoRA model"""
    logger.info(f"Received training request for project {project_id}")
    logger.debug(f"Payload received: baseModelName={payload.baseModelName}, repeats={payload.repeats}")

    try:
        # 1. Get Project Details
        project = await get_project_details(project_id)

        # 2. Determine target directory structure details
        repeats = payload.repeats
        # Sanitize project name for folder name, or use a default/type
        class_name = re.sub(r'\W+', '', project.name) if project.name else 'instance' 
        if not class_name: class_name = "instance" # Ensure it's not empty

        source_dir = get_project_image_path(project_id) # Usually data/datasets/{project_id}/images
        logger.info(f"Using image source directory: {source_dir}")
        
        # --- Logic to move files is REMOVED --- 
        # The TrainingService will now handle creating temporary symlinks instead.

        # 3. Prepare Settings for Training Service
        kohya_settings = payload.settings
        kohya_settings['train_data_dir'] = source_dir # Kohya needs the parent directory
        
        # Pass repeats and class_name into settings for service layer to use and remove
        kohya_settings['repeats'] = repeats 
        kohya_settings['class_name'] = class_name 
        
        # Add baseModelName to settings for the training service to use and remove
        kohya_settings['baseModelName'] = payload.baseModelName

        # 4. Start Training via Service
        logger.info(f"Calling training service for project {project_id} with model {payload.baseModelName}")
        process_info = training_service.start_training(
            project_id=project_id,
            settings=kohya_settings # Pass the full settings dict
            # The service needs to know how to get the base model path from kohya_settings['pretrained_model_name_or_path'] or handle baseModelName separately if needed.
        )
        
        # TODO: Improve process tracking beyond simple return
        
        logger.info(f"Training process initiated successfully for project {project_id}")
        return {"status": "training_started", "details": f"Dataset prepared in {source_dir}", "process_info": process_info}

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception during training start for {project_id}: {http_exc.detail}")
        raise http_exc # Re-raise FastAPI errors
    except Exception as e:
        logger.error(f"Error starting training for project {project_id}: {e}", exc_info=True) # Log traceback
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/{project_id}/training-status")
async def get_training_status(project_id: str):
    """Get the current training status"""
    try:
        status = training_service.get_training_status(project_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/auto-tag")
async def auto_tag_images(project_id: str, settings: AutoTagSettings):
    """Auto-tag all images in a project using WD14 tagger"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting auto-tag process for project {project_id}")
    try:
        # Get project details to fetch trigger word
        project = await get_project_details(project_id)
        trigger_word = project.triggerWord.strip() if project.triggerWord else None
        if trigger_word:
             logger.info(f"Using trigger word: {trigger_word}")
        
        logger.debug(f"Auto-tag settings: {settings.model_dump()}")
        image_dir = get_project_image_path(project_id)
        logger.info(f"Scanning image directory: {image_dir}")
        
        if not os.path.exists(image_dir):
            logger.error(f"Image directory not found: {image_dir}")
            raise HTTPException(status_code=404, detail="Project images directory not found")
            
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Found {len(images)} images to process")
        
        # Load existing tags if needed
        existing_tags = {}
        if settings.existing_tags_mode != "overwrite":
            for img_file in images:
                txt_path = os.path.join(image_dir, os.path.splitext(img_file)[0] + ".txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r") as f:
                        content = f.read().strip()
                        if content:
                            existing_tags[img_file] = [t.strip() for t in content.split(",") if t.strip()]
            logger.info(f"Loaded {len(existing_tags)} existing tag entries")
        
        results = {}
        for idx, image_name in enumerate(images, 1):
            logger.info(f"Processing image {idx}/{len(images)}: {image_name}")
            try:
                image_path = os.path.join(image_dir, image_name)
                
                # Get base tags based on label type
                if settings.label_type == "caption":
                    logger.debug(f"Using BLIP caption mode for {image_name}")
                    base_tags = tagger.get_blip_caption(image_path)
                else:  # tag mode
                    logger.debug(f"Using WD14 tag mode for {image_name}")
                    base_tags = tagger.get_wd14_tags(
                        image_path, 
                        general_threshold=settings.min_threshold,
                        character_threshold=settings.min_threshold,
                        max_tags=settings.max_tags
                    )
                    base_tags = [tag for tag in base_tags if tag not in settings.blacklist_tags]
                
                logger.debug(f"Generated {len(base_tags)} base tags for {image_name}")
                
                # Combine tags based on settings
                final_tags = []
                
                # --- Prepend Trigger Word (if exists) --- 
                if trigger_word:
                    final_tags.append(trigger_word)
                # --- End Prepend Trigger Word ---
                
                # Add prepend tags (from settings)
                final_tags.extend(settings.prepend_tags)
                
                # Add existing tags if in append mode
                if settings.existing_tags_mode == "append" and image_name in existing_tags:
                    final_tags.extend(existing_tags[image_name])
                
                # Add new tags
                final_tags.extend(base_tags)
                
                # Add append tags
                final_tags.extend(settings.append_tags)
                
                # Remove duplicates while preserving order (trigger word priority)
                seen = set()
                final_tags_unique = []
                for tag in final_tags:
                    if tag not in seen:
                        final_tags_unique.append(tag)
                        seen.add(tag)
                
                final_tags = final_tags_unique # Use the unique list
                logger.debug(f"Final tags for {image_name}: {final_tags}")
                
                # Save to individual file
                txt_path = os.path.splitext(image_path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(", ".join(final_tags))
                
                # Store for response
                results[image_name] = final_tags
                
            except Exception as e:
                logger.error(f"Error processing {image_name}: {str(e)}")
                results[image_name] = []
        
        logger.info("Auto-tag process completed successfully")
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Auto-tag process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/images/{project_id}")
async def get_project_images(project_id: str) -> Dict[str, List[str]]:
    try:
        image_dir = get_project_image_path(project_id)
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/auto-tag/{project_id}/{image_name}")
async def auto_tag_image(project_id: str, image_name: str, mode: str = "clip") -> List[str]:
    try:
        image_path = os.path.join(get_project_image_path(project_id), image_name)
        
        if mode == "clip":
            # Load and preprocess image
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
            
            # Predefined categories for CLIP
            categories = [
                "person", "landscape", "building", "animal", "vehicle",
                "food", "plant", "indoor", "outdoor", "abstract",
                "artistic", "realistic", "colorful", "black and white",
                "portrait", "action", "still life", "nature", "urban",
                "night", "day", "water", "sky", "text", "technology"
            ]
            
            # Get text features
            text_inputs = processor(text=categories, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                # Get image and text features
                image_features = model.get_image_features(**inputs)
                text_features = model.get_text_features(**text_inputs)
                
                # Calculate similarity scores
                similarity = torch.nn.functional.cosine_similarity(
                    image_features[:, None], 
                    text_features[None, :], 
                    dim=-1
                )
                
                # Get top 5 matches
                values, indices = similarity[0].topk(5)
                
                # Return categories with confidence > 0.2
                return [categories[i] for i, v in zip(indices, values) if v > 0.2]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tags/{project_id}/{image_name}")
async def save_image_tags(
    project_id: str,
    image_name: str,
    data: Dict[str, List[str]]
) -> Dict[str, str]:
    try:
        # Save tags to a file or database
        tags_file = os.path.join(get_project_image_path(project_id), "tags.txt")
        with open(tags_file, "a") as f:
            f.write(f"{image_name}: {','.join(data['tags'])}\n")
        return {"message": "Tags saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_training_templates() -> Dict[str, Any]:
    """Get available training templates"""
    try:
        templates = {}
        template_dir = "data/templates"
        
        # Read all template files
        for filename in os.listdir(template_dir):
            if filename.endswith('.json'):
                with open(os.path.join(template_dir, filename), 'r') as f:
                    template_name = os.path.splitext(filename)[0]
                    templates[template_name] = json.load(f)
        
        return {"templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/base-models")
async def get_base_models() -> Dict[str, List[str]]:
    """Get available base models for training by scanning the models directory."""
    models_dir = os.path.join("data", "models")
    available_models = []
    try:
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.lower().endswith('.safetensors'):
                    # Use filename without extension as the model name
                    model_name = os.path.splitext(filename)[0]
                    available_models.append(model_name)
        else:
            logger.warning(f"Models directory not found: {models_dir}")
            
        # Sort for consistent order
        available_models.sort()
        
        logger.info(f"Found base models: {available_models}")
        return {"models": available_models}
        
    except Exception as e:
        logger.error(f"Error scanning base models directory {models_dir}: {e}", exc_info=True)
        # Return empty list or raise an error depending on desired behavior
        # Raising might be better to alert the frontend that something is wrong
        raise HTTPException(status_code=500, detail="Could not retrieve base models.")

@router.get("/template-for-model")
async def get_template_for_model(model: str, type: LoraType) -> Dict[str, Any]:
    """Get the appropriate template based on model and LoRA type"""
    try:
        template_map = {
            "IllustriousXL": {
                "character": "illustriousxl_char",
                "style": "illustriousxl_style"
            },
            "PonyXL": {
                "character": "ponyxl_char",
                "style": "ponyxl_style"  
            },
            "SDXL": {
                "character": "sdxl_char",
                "style": "sdxl_style"
            }
        }
        
        if model not in template_map:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
            
        template_name = template_map[model][type]
        template_path = f"data/templates/{template_name}.json"
        
        if not os.path.exists(template_path):
            raise HTTPException(status_code=404, detail=f"Template not found: {template_name}")
            
        with open(template_path, 'r') as f:
            template = json.load(f)
            
        return {
            "template_name": template_name,
            "template": template
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- New Log Endpoint --- 
@router.get("/{project_id}/logs", response_class=PlainTextResponse)
async def get_training_logs(project_id: str, log_type: str = "stdout", lines: int = 100):
    """Retrieve the last N lines from training log files."""
    logger.debug(f"Request received for {log_type} logs for project {project_id}, last {lines} lines.")
    if log_type not in ["stdout", "stderr"]:
        raise HTTPException(status_code=400, detail="Invalid log_type. Must be 'stdout' or 'stderr'.")

    # Construct log file path
    # Note: Assumes standard log directory structure used by TrainingService
    log_dir = os.path.join("data", "datasets", project_id, "log")
    log_filename = f"training_{log_type}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger.debug(f"Attempting to read log file: {log_filepath}")

    if not os.path.exists(log_filepath):
        logger.warning(f"Log file not found: {log_filepath}")
        return f"Log file ({log_filename}) not found."
        # Alternative: raise HTTPException(status_code=404, detail=f"Log file ({log_filename}) not found.")

    try:
        # Read the last N lines (more efficient ways exist for huge files, but ok for now)
        with open(log_filepath, "r", encoding="utf-8") as f:
            # Read all lines
            all_lines = f.readlines()
            # Get the last N lines
            last_n_lines = all_lines[-lines:]
            log_content = "".join(last_n_lines)
            
        logger.debug(f"Successfully read {len(last_n_lines)} lines from {log_filepath}")
        return log_content
    except Exception as e:
        logger.error(f"Error reading log file {log_filepath}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

@router.post("/{project_id}/cancel")
async def cancel_training(project_id: str):
    """Endpoint to cancel an ongoing training process."""
    try:
        result = training_service.cancel_training(project_id)
        return result
    except HTTPException as http_exc:
        # Log FastAPI specific errors differently if needed, or just re-raise
        logger.error(f"HTTP Exception during training cancel for {project_id}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Error cancelling training for project {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel training: {str(e)}") 