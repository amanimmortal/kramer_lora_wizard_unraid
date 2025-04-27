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
import random # Added for sampling tags
import tempfile # Added for temporary prompt file

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

        # --- Auto-generate sample prompt if empty ---
        sample_prompt_setting = kohya_settings.get('sample_prompts', '')
        if not sample_prompt_setting or not str(sample_prompt_setting).strip():
            logger.info("Sample prompt is empty, attempting to auto-generate...")
            trigger_word = project.triggerWord.strip() if project.triggerWord else None
            if trigger_word:
                all_tags = set()
                image_dir = kohya_settings.get('train_data_dir')
                if image_dir and os.path.isdir(image_dir):
                    for filename in os.listdir(image_dir):
                        if filename.lower().endswith('.txt'):
                            try:
                                txt_path = os.path.join(image_dir, filename)
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    content = f.read().strip()
                                    if content:
                                        tags_in_file = {t.strip() for t in content.split(',') if t.strip() and t.strip().lower() != trigger_word.lower()}
                                        all_tags.update(tags_in_file)
                            except Exception as e:
                                logger.warning(f"Could not read or parse tag file {filename}: {e}")
                
                unique_tags = list(all_tags)
                num_tags_to_sample = min(7, len(unique_tags))
                sampled_tags = random.sample(unique_tags, num_tags_to_sample)
                
                generated_prompt = trigger_word
                if sampled_tags:
                    generated_prompt += ", " + ", ".join(sampled_tags)
                
                kohya_settings['sample_prompts'] = generated_prompt
                logger.info(f"Auto-generated sample prompt: {generated_prompt}")
            else:
                logger.warning("Cannot auto-generate sample prompt: No trigger word set for the project.")
        # --- End auto-generate --- 

        # --- Handle Sample Prompt File ---
        sample_prompt_value = kohya_settings.get('sample_prompts')
        temp_prompt_file_path = None # Track if we create a temp file

        if sample_prompt_value and isinstance(sample_prompt_value, str) and sample_prompt_value.strip():
            # If sample_prompts is a non-empty string, save it to a file
            try:
                metadata_dir = os.path.join("data", "datasets", project_id, "metadata")
                os.makedirs(metadata_dir, exist_ok=True) # Ensure metadata dir exists

                # Use a predictable name for easier cleanup later
                temp_prompt_filename = f"temp_sample_prompt_{project_id}.txt"
                temp_prompt_file_path = os.path.join(metadata_dir, temp_prompt_filename)

                with open(temp_prompt_file_path, 'w', encoding='utf-8') as f:
                    f.write(sample_prompt_value)

                # Update settings to use the file path
                kohya_settings['sample_prompts'] = temp_prompt_file_path
                logger.info(f"Saved sample prompt string to temporary file: {temp_prompt_file_path}")

            except Exception as e:
                 logger.error(f"Failed to create temporary sample prompt file: {e}", exc_info=True)
                 # Decide how to handle: Proceed without sampling or raise error?
                 # For now, let's remove the setting so sampling is skipped if file creation fails.
                 if 'sample_prompts' in kohya_settings:
                     del kohya_settings['sample_prompts']
                 logger.warning("Proceeding without sample prompt file due to error.")
                 # Optionally, raise HTTPException here if sampling is critical

        # --- End Handle Sample Prompt File ---

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
    """Fetches available training templates from the data directory."""
    logger.info("Fetching training templates...")
    # Use absolute path for clarity in logs
    template_dir = os.path.abspath("data/templates")
    logger.debug(f"Attempting to access template directory: {template_dir}")
    templates = {}
    try:
        # Check directory existence and accessibility
        logger.debug(f"Checking if path exists and is a directory: {template_dir}")
        if not os.path.isdir(template_dir):
            logger.error(f"Template directory check failed. Path not found or not a directory: {template_dir}")
            # Add extra info if possible
            exists = os.path.exists(template_dir)
            logger.error(f"os.path.exists reports: {exists}")
            if exists:
                 logger.error(f"Path exists but is not a directory.")
            raise HTTPException(status_code=500, detail=f"Template directory not found or inaccessible: {template_dir}")
        logger.debug(f"Template directory confirmed to exist and is a directory: {template_dir}")

        # List directory contents
        logger.debug(f"Attempting to list contents of: {template_dir}")
        try:
             filenames = os.listdir(template_dir)
             logger.info(f"Successfully listed directory. Found items: {filenames}") # Log items found
        except PermissionError as pe:
             logger.error(f"Permission denied when trying to list directory contents for {template_dir}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Permission denied listing template directory: {template_dir}")
        except Exception as list_e:
             logger.error(f"Failed to list directory contents for {template_dir}: {list_e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Failed to list template directory: {template_dir}")

        if not filenames:
             logger.warning(f"Template directory is empty: {template_dir}")

        for filename in filenames:
            logger.debug(f"Processing item: {filename}")
            file_path = os.path.join(template_dir, filename)
            # Check if it's a file and ends with .json
            if not os.path.isfile(file_path):
                 logger.debug(f"Skipping item '{filename}', not a file.")
                 continue
            if not filename.lower().endswith(".json"): # Case-insensitive check
                 logger.debug(f"Skipping item '{filename}', not a .json file.")
                 continue

            logger.debug(f"Attempting to read JSON template file: {file_path}")
            try:
                # Explicitly use utf-8 encoding
                with open(file_path, "r", encoding="utf-8") as f:
                    template_data = json.load(f)
                    template_key = os.path.splitext(filename)[0]
                    templates[template_key] = template_data
                logger.debug(f"Successfully loaded template: {filename}")
            except json.JSONDecodeError as json_err:
                # Log the specific JSON error and the file
                logger.error(f"Error decoding JSON from template file: {filename}. Error: {json_err}", exc_info=True)
                # Decide if you want to continue or fail hard; currently continues to next file
            except FileNotFoundError:
                 logger.error(f"File not found during read attempt (race condition?): {file_path}", exc_info=True)
            except PermissionError:
                 logger.error(f"Permission denied when trying to read file: {file_path}", exc_info=True)
            except Exception as read_e:
                # Catch other potential file reading errors
                logger.error(f"Unexpected error reading or processing template file {filename}: {read_e}", exc_info=True)

        logger.info(f"Finished processing directory. Found {len(templates)} valid templates.")
        if not templates and filenames: # Log if files were present but none loaded
            logger.warning("Directory contained files, but no valid JSON templates were loaded.")
        elif not templates and not filenames:
             logger.info("No templates loaded as the directory was empty.")


        return templates
    except HTTPException as http_exc: # Re-raise HTTP exceptions directly
        logger.error(f"HTTPException occurred during template loading: {http_exc.detail}", exc_info=True)
        raise http_exc
    except Exception as e:
        # This catches errors from the initial checks or unforeseen issues
        logger.error(f"Unhandled exception while trying to load templates from {template_dir}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load templates due to an unexpected error: {str(e)}")

@router.get("/base-models")
async def get_base_models() -> Dict[str, List[str]]:
    """Get available base models for training by scanning the models directory."""
    models_dir = os.path.join("data", "models")
    available_models = []
    logger.info(f"Fetching base models from: {models_dir}") # Added logging
    try:
        if os.path.isdir(models_dir): # Check if it's a directory
            logger.debug(f"Scanning directory: {models_dir}")
            for filename in os.listdir(models_dir):
                # Check if it's a file and has the correct extension
                file_path = os.path.join(models_dir, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.ckpt', '.safetensors')):
                    # Use filename without extension as the model name
                    model_name = os.path.splitext(filename)[0]
                    logger.debug(f"Found model file: {filename}, adding as: {model_name}")
                    available_models.append(model_name)
                # else: # Optional: Log skipped files/dirs
                #    logger.debug(f"Skipping item: {filename}")
        else:
            logger.warning(f"Base models directory not found or is not a directory: {models_dir}")

        # Sort for consistent order
        available_models.sort()

        logger.info(f"Found base models: {available_models}")
        return {"models": available_models}

    except PermissionError as pe:
        logger.error(f"Permission denied accessing base models directory {models_dir}", exc_info=True)
        raise HTTPException(status_code=500, detail="Permission denied accessing base models directory.")
    except Exception as e:
        logger.error(f"Error scanning base models directory {models_dir}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve base models.")

@router.get("/template-for-model")
async def get_template_for_model(model: str, type: LoraType) -> Dict[str, Any]:
    """
    Attempts to find a suitable JSON template based on the model name and LoRA type (char/style).
    Looks for files named like: {model_name_part}_{type}.json 
    Example: ponyxl_char.json, illustriousxl_style.json
    """
    logger.info(f"Attempting to find template for model '{model}' and type '{type.value}'")
    template_dir = os.path.abspath("data/templates")
    
    # --- Derive potential template filename ---
    # This logic relies on the flat model name provided by the reverted get_base_models
    # Example: model = "ponyDiffusionV6XL_v6StartWithThisOne"
    model_name_part = model # Use the provided model name directly
    
    # Construct expected filename
    template_filename = f"{model_name_part.lower()}_{type.value.lower()}.json"
    potential_path = os.path.join(template_dir, template_filename)
    logger.debug(f"Looking for template file: {potential_path} based on model name '{model}'")

    # --- Attempt to load the specific template ---
    if os.path.isfile(potential_path):
        logger.info(f"Found matching template file: {template_filename}")
        try:
            with open(potential_path, "r", encoding="utf-8") as f:
                template_data = json.load(f)
            logger.debug(f"Successfully loaded template data for {template_filename}")
            return template_data
        except json.JSONDecodeError as json_err:
            logger.error(f"Error decoding JSON from specific template file: {template_filename}. Error: {json_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to parse template file '{template_filename}'.")
        except PermissionError:
             logger.error(f"Permission denied reading template file: {potential_path}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Permission denied reading template '{template_filename}'.")
        except Exception as e:
            logger.error(f"Error reading specific template file {template_filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to read template file '{template_filename}'.")
    else:
        # --- Fallback logic (Optional) ---
        fallback_filename = f"sdxl_{type.value.lower()}.json" # Example fallback
        fallback_path = os.path.join(template_dir, fallback_filename)
        logger.warning(f"Specific template '{template_filename}' not found. Attempting fallback: '{fallback_filename}'")
        
        if os.path.isfile(fallback_path):
            logger.info(f"Found fallback template file: {fallback_filename}")
            try:
                with open(fallback_path, "r", encoding="utf-8") as f:
                    fallback_data = json.load(f)
                logger.debug(f"Successfully loaded fallback template data for {fallback_filename}")
                return fallback_data
            except Exception as e:
                 logger.error(f"Error reading fallback template file {fallback_filename}: {e}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Failed to read fallback template file '{fallback_filename}'.")
        else:
             logger.error(f"Specific template '{template_filename}' and fallback '{fallback_filename}' not found in {template_dir}")
             raise HTTPException(status_code=404, detail=f"Could not find a suitable template for model '{model}' type '{type.value}'. Looked for '{template_filename}' and fallback '{fallback_filename}'.")

# --- New Log Endpoint --- 
@router.get("/{project_id}/logs", response_class=PlainTextResponse)
async def get_training_logs(project_id: str, log_type: str = "stdout", lines: int = 100):
    """Get training logs"""
    logger.debug(f"Fetching logs for project {project_id}, type: {log_type}, lines: {lines}")
    try:
        log_content = training_service.get_training_logs(project_id, log_type, lines)
        return PlainTextResponse(log_content)
    except FileNotFoundError:
        logger.warning(f"Log file not found for project {project_id}, type {log_type}")
        raise HTTPException(status_code=404, detail="Log file not found")
    except Exception as e:
        logger.error(f"Error fetching logs for project {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/cancel")
async def cancel_training(project_id: str):
    """Cancel an ongoing training process"""
    logger.info(f"Received request to cancel training for project {project_id}")
    try:
        result = training_service.cancel_training(project_id)
        if result["status"] == "cancelled":
            logger.info(f"Training cancelled successfully for project {project_id}")
            return result
        elif result["status"] == "not_running":
             logger.warning(f"Attempted to cancel training for {project_id}, but no process was found.")
             raise HTTPException(status_code=404, detail="Training process not found or already finished")
        else:
             logger.error(f"Failed to cancel training for project {project_id}: {result.get('details', 'Unknown reason')}")
             raise HTTPException(status_code=500, detail=result.get("details", "Failed to cancel training"))
    except Exception as e:
        logger.error(f"Error cancelling training for project {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))