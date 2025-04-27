from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from typing import List, Optional
import os
import uuid
from datetime import datetime
import shutil
import json
import logging
from PIL import Image
from pydantic import BaseModel

from ..models.lora import LoraProject, LoraType, ImageMetadata, ImageTag, TrainingSettings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["projects"])

class ProjectCreate(BaseModel):
    name: str
    type: LoraType

class TriggerWordUpdate(BaseModel):
    triggerWord: Optional[str] = None

def create_project_structure(project_id: str):
    """Create the directory structure for a new project following Kohya_ss requirements"""
    base_path = f"data/datasets/{project_id}"
    
    # Main directories
    os.makedirs(f"{base_path}/images", exist_ok=True)  # For training images
    os.makedirs(f"{base_path}/metadata", exist_ok=True)  # For project metadata
    os.makedirs(f"{base_path}/reg", exist_ok=True)  # For regularization images
    os.makedirs(f"{base_path}/log", exist_ok=True)  # For training logs
    os.makedirs(f"{base_path}/output", exist_ok=True)  # For training outputs
    
    # Create project state file
    state = {
        "current_step": "created",
        "training_settings": None,
        "last_modified": datetime.utcnow().isoformat()
    }
    with open(f"{base_path}/metadata/state.json", "w") as f:
        json.dump(state, f)
    
    return base_path

@router.post("/", response_model=LoraProject)
async def create_project(project: ProjectCreate):
    """Create a new LoRA project"""
    logger.debug(f"Creating new project with data: {project}")
    try:
        project_id = str(uuid.uuid4())
        project_data = LoraProject(
            id=project_id,
            name=project.name,
            type=project.type,
            created_at=datetime.utcnow().isoformat(),
            images=[]
        )
        
        # Create project directory structure
        create_project_structure(project_id)
        
        # Save project metadata
        with open(f"data/datasets/{project_id}/metadata/project.json", "w") as f:
            f.write(project_data.model_dump_json())
        
        logger.info(f"Successfully created project {project_id}")
        return project_data
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/images", response_model=ImageMetadata)
async def upload_image(project_id: str, file: UploadFile = File(...)):
    """Upload an image to a project"""
    try:
        # Verify project exists
        if not os.path.exists(f"data/datasets/{project_id}"):
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Save image
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        file_path = f"data/datasets/{project_id}/images/{filename}"
        
        file_size = 0
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            file_size = buffer.tell() # Get file size after writing
        
        # Get image dimensions if possible
        width, height = None, None
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except Exception as img_err:
            logger.warning(f"Could not read image dimensions for {filename}: {img_err}")

        metadata = ImageMetadata(
            id=filename, 
            filename=filename,
            tags=[],
            caption="",
            width=width,
            height=height,
            file_size=file_size
        )
        
        # Save metadata to its own JSON file
        metadata_file_path = f"data/datasets/{project_id}/metadata/{filename}.json"
        with open(metadata_file_path, "w") as f:
            f.write(metadata.model_dump_json())
            
        # Add image metadata reference to the main project file
        project_file_path = f"data/datasets/{project_id}/metadata/project.json"
        if os.path.exists(project_file_path):
             with open(project_file_path, "r+") as f:
                project_data = json.load(f)
                if 'images' not in project_data:
                    project_data['images'] = []
                # Append the metadata
                project_data['images'].append(metadata.model_dump()) 
                f.seek(0)
                json.dump(project_data, f)
                f.truncate()

        # Create empty tag file (Kohya_ss requirement)
        txt_path = f"data/datasets/{project_id}/images/{os.path.splitext(filename)[0]}.txt"
        with open(txt_path, "w") as f:
            f.write("")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/images/{image_filename}/tags")
async def update_image_tags(project_id: str, image_filename: str, tags_to_add: List[ImageTag] = Body(...)):
    """Update tags for an image by adding new tags (ensuring no duplicates)."""
    try:
        metadata_path = f"data/datasets/{project_id}/metadata/{image_filename}.json"
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")
            raise HTTPException(status_code=404, detail="Image metadata not found")
        
        # Read existing metadata
        with open(metadata_path, "r+") as f:
            try:
                metadata = ImageMetadata.model_validate_json(f.read())
            except Exception as json_err:
                 logger.error(f"Error reading metadata JSON for {image_filename}: {json_err}")
                 raise HTTPException(status_code=500, detail="Could not read image metadata")

            # Ensure metadata.tags is a list
            if not isinstance(metadata.tags, list):
                logger.warning(f"Tags field for {image_filename} is not a list, resetting.")
                metadata.tags = []

            # Add new tags, avoiding duplicates
            current_tags_set = set(metadata.tags)
            added_count = 0
            for new_tag_obj in tags_to_add:
                new_tag = new_tag_obj.tag.strip()
                if new_tag and new_tag not in current_tags_set:
                    metadata.tags.append(new_tag)
                    current_tags_set.add(new_tag)
                    added_count += 1
            
            logger.debug(f"Added {added_count} new tags to {image_filename}. Current tags: {metadata.tags}")

            # Save updated metadata
            f.seek(0)
            f.write(metadata.model_dump_json(indent=2)) # Add indent for readability
            f.truncate()
        
        # Update Kohya_ss tag file
        txt_path = f"data/datasets/{project_id}/images/{os.path.splitext(image_filename)[0]}.txt"
        try:
            with open(txt_path, "w") as f:
                f.write(", ".join(metadata.tags))
        except Exception as txt_err:
             logger.error(f"Error writing tag file {txt_path}: {txt_err}")
             # Don't fail the whole request if this minor step fails
        
        return metadata
        
    except HTTPException as http_exc: # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Error updating tags for {image_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error updating tags: {str(e)}")

@router.get("/{project_id}", response_model=LoraProject)
async def get_project(project_id: str):
    """Get project details including image metadata and current state"""
    try:
        project_file_path = f"data/datasets/{project_id}/metadata/project.json"
        metadata_dir = f"data/datasets/{project_id}/metadata"
        images_dir = f"data/datasets/{project_id}/images"

        if not os.path.exists(project_file_path):
            raise HTTPException(status_code=404, detail="Project not found")

        # Read base project metadata
        with open(project_file_path, "r") as f:
            project = LoraProject.model_validate_json(f.read())

        # Load individual image metadata files
        project.images = []
        # --- Load trigger word (new) ---
        project_data_dict = project.model_dump() # Get dict to safely check triggerWord
        project.triggerWord = project_data_dict.get('triggerWord')
        # --- End Load trigger word ---

        for filename in os.listdir(metadata_dir):
            if filename.endswith('.json') and filename not in ["project.json", "state.json"]:
                try:
                    with open(os.path.join(metadata_dir, filename), "r") as img_f:
                        image_meta = ImageMetadata.model_validate_json(img_f.read())
                        
                        # Read tags from individual txt file
                        txt_path = os.path.join(images_dir, os.path.splitext(image_meta.filename)[0] + ".txt")
                        if os.path.exists(txt_path):
                            with open(txt_path, "r", encoding="utf-8") as txt_f:
                                content = txt_f.read().strip()
                                if content:  # Only split if there's content
                                    image_meta.tags = [t.strip() for t in content.split(",") if t.strip()]
                                else:
                                    image_meta.tags = []
                        
                        project.images.append(image_meta)
                except Exception as e:
                    logger.warning(f"Could not load metadata for {filename}: {e}")

        # Read current state
        state_path = os.path.join(metadata_dir, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
                project.training_status = state.get("current_step")
                project.last_modified = state.get("last_modified")

        return project
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    except Exception as e:
        logger.error(f"Error getting project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/trigger-word", response_model=LoraProject)
async def update_project_trigger_word(project_id: str, update_data: TriggerWordUpdate):
    """Update the trigger word for a project."""
    project_file_path = f"data/datasets/{project_id}/metadata/project.json"
    if not os.path.exists(project_file_path):
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        with open(project_file_path, "r+") as f:
            project_data = json.load(f) # Load as dict
            
            old_trigger_word = project_data.get('triggerWord')
            new_trigger_word = update_data.triggerWord.strip() if update_data.triggerWord else None
            
            if old_trigger_word == new_trigger_word:
                # No change needed, return current data
                 project = LoraProject.model_validate(project_data)
                 return project # Return validated model
                 
            project_data['triggerWord'] = new_trigger_word
            project_data['last_modified'] = datetime.utcnow().isoformat()

            f.seek(0)
            json.dump(project_data, f, indent=2) # Write back as dict, add indent
            f.truncate()
        
        logger.info(f"Updated trigger word for project {project_id} to: {new_trigger_word}")
        # Return the updated project model
        project = LoraProject.model_validate(project_data)
        return project
    except Exception as e:
        logger.error(f"Error updating trigger word for project {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update trigger word: {str(e)}")

@router.post("/{project_id}/training-settings")
async def update_training_settings(project_id: str, settings: TrainingSettings):
    """Update training settings and save to project state"""
    try:
        state_path = f"data/datasets/{project_id}/metadata/state.json"
        if not os.path.exists(state_path):
            raise HTTPException(status_code=404, detail="Project not found")
        
        with open(state_path, "r") as f:
            state = json.load(f)
        
        state["training_settings"] = settings.model_dump()
        state["last_modified"] = datetime.utcnow().isoformat()
        
        with open(state_path, "w") as f:
            json.dump(state, f)
        
        return {"message": "Training settings updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[LoraProject])
async def list_projects():
    """List all projects with their current state and image count"""
    try:
        projects = []
        datasets_dir = "data/datasets"
        
        if os.path.exists(datasets_dir):
            for project_id in os.listdir(datasets_dir):
                project_path = os.path.join(datasets_dir, project_id)
                metadata_dir = os.path.join(project_path, "metadata")
                project_file_path = os.path.join(metadata_dir, "project.json")
                
                if os.path.isdir(project_path) and os.path.exists(project_file_path):
                    try:
                        # Read project metadata
                        with open(project_file_path, "r") as f:
                            project = LoraProject.model_validate_json(f.read())
                        
                        # Load image metadata just to count them
                        project.images = [] # Reset images list from project.json
                        for filename in os.listdir(metadata_dir):
                            if filename.endswith('.json') and filename not in ["project.json", "state.json"]:
                                 project.images.append(ImageMetadata(id=filename, filename=filename, tags=[], caption="")) # Dummy object for count

                        # Read current state
                        state_path = os.path.join(metadata_dir, "state.json")
                        if os.path.exists(state_path):
                            with open(state_path, "r") as f:
                                state = json.load(f)
                                project.training_status = state.get("current_step")
                                project.training_progress = state.get("training_progress")
                                project.last_modified = state.get("last_modified")
                        
                        projects.append(project)
                    except Exception as e:
                        logger.error(f"Error loading project {project_id}: {str(e)}")
                        continue
        
        # Sort projects by last modified date, most recent first
        projects.sort(key=lambda x: x.last_modified or x.created_at, reverse=True)
        return projects
        
    except Exception as e:
        logger.error(f"Error listing projects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str):
    """Delete a project and all its associated data."""
    project_path = f"data/datasets/{project_id}"
    logger.info(f"Attempting to delete project: {project_id} at path: {project_path}")

    if not os.path.exists(project_path) or not os.path.isdir(project_path):
        logger.warning(f"Project not found for deletion: {project_id}")
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        shutil.rmtree(project_path)
        logger.info(f"Successfully deleted project directory: {project_path}")
        return # Return No Content
    except Exception as e:
        logger.error(f"Error deleting project {project_id}: {str(e)}")
        # Raise a 500 error if deletion fails
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

# --- New Endpoint to Finalize Tags ---
@router.post("/{project_id}/finalize-tags", status_code=200)
async def finalize_project_tags(project_id: str):
    """
    Ensures the project's trigger word is prepended to each image's tag file 
    if it's not already present. Creates the tag file if it doesn't exist.
    Intended to be called before starting training setup.
    """
    logger.info(f"Finalizing tags for project {project_id}, ensuring trigger word.")
    
    try:
        # 1. Get Project Details (includes trigger word)
        project_file_path = f"data/datasets/{project_id}/metadata/project.json"
        if not os.path.exists(project_file_path):
            logger.error(f"Finalize tags: Project metadata file not found: {project_file_path}")
            raise HTTPException(status_code=404, detail="Project definition not found")
            
        with open(project_file_path, "r") as f:
            project_data = json.load(f) # Load as dict to easily get triggerWord
            
        trigger_word = project_data.get('triggerWord')
        if not trigger_word or not trigger_word.strip():
            logger.info(f"No trigger word set for project {project_id}, skipping tag finalization.")
            return {"message": "No trigger word set, skipping."}
        
        trigger_word = trigger_word.strip() # Use the stripped version
        logger.info(f"Using trigger word: '{trigger_word}'")

        # 2. Get Image Directory Path
        image_dir = f"data/datasets/{project_id}/images"
        if not os.path.isdir(image_dir):
             logger.error(f"Finalize tags: Image directory not found: {image_dir}")
             # If image dir doesn't exist, we can't do anything, but maybe not a server error?
             # Let's return a specific message.
             return {"message": f"Image directory not found, cannot finalize tags."}

        # 3. Iterate through images and update tag files
        updated_count = 0
        created_count = 0
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]

        logger.debug(f"Found {len(image_files)} image files to check tags for.")

        for image_filename in image_files:
            base_name = os.path.splitext(image_filename)[0]
            txt_path = os.path.join(image_dir, f"{base_name}.txt")
            
            current_tags = []
            file_existed = False
            
            try:
                if os.path.exists(txt_path):
                    file_existed = True
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            current_tags = [t.strip() for t in content.split(',') if t.strip()]
                
                # Check if trigger word needs adding (case-insensitive)
                trigger_present = any(tag.lower() == trigger_word.lower() for tag in current_tags)
                
                if not trigger_present:
                    # Prepend the trigger word
                    final_tags = [trigger_word] + [tag for tag in current_tags if tag.lower() != trigger_word.lower()] # Ensure no duplicates if case differs
                    tag_string = ", ".join(final_tags)
                    
                    # Write back to file
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(tag_string)
                        
                    if file_existed:
                        updated_count += 1
                        logger.debug(f"Updated tags for {image_filename}, prepended trigger word.")
                    else:
                        created_count += 1
                        logger.debug(f"Created tag file for {image_filename} with trigger word.")
                # else: # Optional: Log if trigger word was already present
                #    logger.debug(f"Trigger word already present for {image_filename}.")

            except PermissionError:
                 logger.error(f"Permission denied accessing tag file: {txt_path}. Skipping.")
                 # Optionally raise an error or collect files with issues
            except Exception as e:
                logger.error(f"Error processing tag file {txt_path}: {e}. Skipping.")
                # Optionally raise an error or collect files with issues

        logger.info(f"Tag finalization complete for project {project_id}. Files created: {created_count}, Files updated: {updated_count}.")
        return {"message": f"Tag finalization complete. Files created: {created_count}, Files updated: {updated_count}."}

    except HTTPException as http_exc:
        raise http_exc # Re-raise FastAPI specific errors
    except Exception as e:
        logger.error(f"Error finalizing tags for project {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error finalizing tags: {str(e)}")
# --- End New Endpoint --- 