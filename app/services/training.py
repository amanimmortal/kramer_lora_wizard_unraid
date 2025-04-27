import os
import subprocess
import json
import logging
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Helper function to convert settings dict to list of command line args
def config_to_args(config: Dict[str, Any]) -> List[str]:
    args = []
    for key, value in config.items():
        # Skip keys that are not meant for the command line or are handled specially
        if value is None or key in ['template_name', 'base_model']: # Example exclusions
            continue
        
        arg_key = f"--{key}"
        
        if isinstance(value, bool):
            if value:
                args.append(arg_key)
        elif isinstance(value, list):
            # Handle list arguments if needed (e.g., --some_list item1 item2)
            args.extend([arg_key, " ".join(map(str, value))])
        elif key == "optimizer_args":
            # Special handling for optimizer_args - split into individual arguments
            if isinstance(value, str):
                args.extend([arg_key] + value.split())
        else:
            # Handle regular key-value pairs
            args.extend([arg_key, str(value)])
            
    return args

class TrainingService:
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
        self.models_path = os.path.join(base_path, "models")
        self.datasets_path = os.path.join(base_path, "datasets")
        
        # Ensure directories exist
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.datasets_path, exist_ok=True)

        self.active_processes: Dict[str, Dict[str, Any]] = {}

    def start_training(self, project_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Directly starts a training process using the prepared settings dictionary"""
        # --- Check if a process is already running for this project ---
        if project_id in self.active_processes:
            existing_process_info = self.active_processes[project_id]
            existing_pid = existing_process_info.get("process").pid if existing_process_info.get("process") else "unknown"
            logger.warning(f"Training start requested for {project_id}, but process {existing_pid} is already active.")
            raise HTTPException(
                status_code=409, # Conflict
                detail=f"Training process (PID: {existing_pid}) is already running or cleaning up for this project. Please wait or cancel the existing run."
            )
        # --- End check ---
        
        try:
            logger.info(f"Starting training process for project {project_id}")
            logger.debug(f"Using settings: {settings}")
            
            # --- Determine Repeats and Class Name (from API layer - passed in settings) ---
            repeats = settings.pop('repeats', 10) # Remove from settings after getting it
            class_name = settings.pop('class_name', 'instance') # Remove from settings
            target_subdir_name = f"{repeats}_{class_name}"
            source_dir = settings.get('train_data_dir') # This should be /app/data/datasets/{project_id}/images
            
            if not source_dir:
                 raise ValueError("train_data_dir is missing from settings")

            target_subdir_path = os.path.join(source_dir, target_subdir_name)
            logger.info(f"Preparing temporary training structure in: {target_subdir_path}")

            # --- Copy Files Instead of Creating Symlinks ---
            os.makedirs(target_subdir_path, exist_ok=True)
            copied_files_count = 0
            files_in_source = os.listdir(source_dir)
            
            for filename in files_in_source:
                source_file_path = os.path.join(source_dir, filename)
                target_file_path = os.path.join(target_subdir_path, filename)
                
                # Check if it's a file and *not* the target directory itself
                if os.path.isfile(source_file_path) and source_file_path != target_subdir_path:
                    name, ext = os.path.splitext(filename)
                    ext_lower = ext.lower()
                    
                    if ext_lower in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.txt']:
                        # Remove any existing file/link at the target
                        if os.path.exists(target_file_path):
                            os.remove(target_file_path)
                        try:
                            shutil.copy2(source_file_path, target_file_path)
                            if ext_lower != '.txt':
                                copied_files_count += 1
                        except Exception as e:
                            logger.error(f"Failed to copy file {source_file_path} to {target_file_path}: {e}")

            logger.info(f"Copied {copied_files_count} image files (and associated .txt) into subfolder for training.")
            if copied_files_count == 0:
                logger.warning(f"No image files found in {source_dir} to copy for training.")
            
            # --- Training Setup (largely the same as before) ---
            
            # Fix mixed precision setting 
            if settings.get("mixed_precision") == "bf16":
                settings["mixed_precision"] = "fp16"
                logger.info("Changed mixed_precision from bf16 to fp16")
            
            # Handle optimizer_args spaces
            if 'optimizer_args' in settings and isinstance(settings['optimizer_args'], str):
                optimizer_args = settings['optimizer_args'].replace(',', ' ').strip()
                settings['optimizer_args'] = optimizer_args
                logger.info(f"Processed optimizer_args: {settings['optimizer_args']}")
            
            # Map model names to file names (This logic seems fine)
            model_filenames = {
                'illustriousxl': 'Illustrious-XL-v1.0',
                'ponyxl': 'ponyDiffusionV6XL_v6StartWithThisOne',
                'sdxl': 'sdxl'
            }
            base_model_path = settings.get('pretrained_model_name_or_path', '')
            base_model = settings.pop('baseModelName', None) # Get baseModelName passed from API
            
            # Defensive check and log
            if not base_model or base_model == "undefined":
                logger.error(f"Invalid base_model received: {base_model}")
                raise ValueError("No valid base model selected. Please select a base model.")

            if base_model and base_model in model_filenames:
                settings["pretrained_model_name_or_path"] = os.path.join("/app/data/models", f"{model_filenames[base_model]}.safetensors")
                logger.info(f"Using model: {model_filenames[base_model]}")
            elif base_model_path:
                 # If baseModelName wasn't provided or didn't match, try extracting from path
                 path_parts = base_model_path.split('/')
                 if len(path_parts) >= 2:
                    filename = path_parts[-1].replace('.safetensors', '').lower()
                    for key, value in model_filenames.items():
                        if value.lower() == filename or key == filename:
                            base_model = key # Set base_model if found via path
                            settings["pretrained_model_name_or_path"] = os.path.join("/app/data/models", f"{model_filenames[base_model]}.safetensors")
                            logger.info(f"Determined base model from path: {model_filenames[base_model]}")
                            break
                 if not base_model:
                    logger.warning(f"Could not determine known base model name from settings/path. Using provided path: {base_model_path}")
                    # Ensure the provided path is used if no mapping occurred
                    settings["pretrained_model_name_or_path"] = base_model_path 
            else:
                 logger.error(f"Critical: No pretrained_model_name_or_path or derivable baseModelName provided.")
                 raise ValueError("Base model path/name is required.")

            # Determine script (This logic seems fine)
            is_sdxl = 'xl' in settings.get('pretrained_model_name_or_path', '').lower()
            script_name = "sdxl_train_network.py" if is_sdxl else "train_network.py"
            script_path = os.path.abspath(script_name) 

            if not os.path.exists(script_path):
                 logger.error(f"Training script not found: {script_path}")
                 raise FileNotFoundError(f"Required training script {script_name} not found.")
            
            logger.info(f"Using training script: {script_name}")

            # Save final config (This logic seems fine)
            config_path = os.path.join(self.datasets_path, project_id, "metadata", "training_config_used.json")
            try:
                 with open(config_path, 'w') as f:
                    json.dump(settings, f, indent=2)
                 logger.info(f"Saved final training config to {config_path}")
            except Exception as write_err:
                 logger.warning(f"Could not save final training config: {write_err}")

            # Build command (This logic is fine, including accelerate flags)
            mixed_precision_setting = settings.get('mixed_precision', 'no')
            launch_command = [
                "accelerate", 
                "launch", 
                "--num_processes=1", 
                "--num_machines=1",
                f"--mixed_precision={mixed_precision_setting}",
                "--dynamo_backend=no", # Explicitly set default
                script_path
            ]
            launch_command.extend(config_to_args(settings)) # Pass settings to script
            
            logger.info(f"Executing command: {' '.join(launch_command)}")
            
            # Start training process (This logic seems fine)
            log_dir = settings.get('logging_dir', os.path.join(self.datasets_path, project_id, "log"))
            stdout_log_path = os.path.join(log_dir, "training_stdout.log")
            stderr_log_path = os.path.join(log_dir, "training_stderr.log")
            os.makedirs(log_dir, exist_ok=True) # Ensure log dir exists

            with open(stdout_log_path, 'w') as stdout_log, open(stderr_log_path, 'w') as stderr_log:
                process = subprocess.Popen(
                    launch_command,
                    stdout=stdout_log, 
                    stderr=stderr_log,
                    text=True, 
                    cwd=os.path.dirname(script_path) # Run from script's directory
                )
            
            pid = process.pid
            # Store process AND the temp dir path for cleanup
            self.active_processes[project_id] = {
                "process": process, 
                "temp_dir": target_subdir_path 
            }
            logger.info(f"Training process started with PID: {pid}")
            
            self._update_project_state(project_id, "training") 
            return {"pid": pid, "log_files": {"stdout": stdout_log_path, "stderr": stderr_log_path}}
            
        except Exception as e:
            logger.error(f"Error during training startup for {project_id}: {e}", exc_info=True)
            # Attempt cleanup if temp dir was created
            if 'target_subdir_path' in locals() and os.path.exists(target_subdir_path):
                 logger.warning(f"Cleaning up temporary directory due to startup error: {target_subdir_path}")
                 try:
                     shutil.rmtree(target_subdir_path)
                 except Exception as cleanup_e:
                     logger.error(f"Error during cleanup of {target_subdir_path}: {cleanup_e}")
            self._update_project_state(project_id, "error", {"error": str(e)}) 
            raise 

    def _cleanup_training_artifacts(self, project_id: str):
        """Removes temporary directories and process tracking."""
        if project_id in self.active_processes:
            process_info = self.active_processes[project_id]
            temp_dir = process_info.get("temp_dir")
            if temp_dir and os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary training directory: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.error(f"Error removing temporary directory {temp_dir}: {e}")

            # --- Clean up temporary sample prompt file --- 
            try:
                metadata_dir = os.path.join(self.datasets_path, project_id, "metadata")
                temp_prompt_filename = f"temp_sample_prompt_{project_id}.txt"
                temp_prompt_file_path = os.path.join(metadata_dir, temp_prompt_filename)
                if os.path.exists(temp_prompt_file_path):
                     logger.info(f"Cleaning up temporary sample prompt file: {temp_prompt_file_path}")
                     os.remove(temp_prompt_file_path)
            except Exception as e:
                 logger.error(f"Error removing temporary sample prompt file for project {project_id}: {e}")
            # --- End cleanup ---
            
            # Remove from active process list
            del self.active_processes[project_id]
            logger.debug(f"Removed project {project_id} from active processes.")
        else:
             logger.debug(f"Cleanup requested for {project_id}, but it was not found in active processes.")

    def _update_project_state(self, project_id: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Update the project's state file"""
        state_path = os.path.join(self.datasets_path, project_id, "metadata", "state.json")
        try:
            state = {}
            if os.path.exists(state_path):
                 with open(state_path, 'r') as f:
                    state = json.load(f)
            
            state["current_step"] = status
            state["last_modified"] = datetime.utcnow().isoformat()
            
            if details:
                state["status_details"] = details # Store error or other details
            elif "status_details" in state:
                 del state["status_details"] # Clear old details if status is not error

            # Keep existing training settings if present
            if "training_settings" not in state:
                 state["training_settings"] = None
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            # Call cleanup if the status marks the end of the process run
            if status in ["completed", "error", "cancelled"]:
                 self._cleanup_training_artifacts(project_id)
            
        except Exception as e:
            logger.error(f"Failed to update project state file {state_path}: {e}")
            # Don't raise here, log and continue if possible
    
    def get_training_status(self, project_id: str) -> Dict[str, Any]:
        """Get current training status and handle process completion cleanup."""
        state = {}
        try:
            state_path = os.path.join(self.datasets_path, project_id, "metadata", "state.json")
            if os.path.exists(state_path):
                 with open(state_path, 'r') as f:
                    state = json.load(f)
            else:
                 return {"status": "unknown", "details": "State file not found."}

            process_entry = self.active_processes.get(project_id)
            if process_entry:
                 process = process_entry.get("process")
                 if not process: # Should not happen if entry exists
                      logger.error(f"Inconsistent state: Process entry found for {project_id} but no process object.")
                      self._cleanup_training_artifacts(project_id) # Clean up inconsistent entry
                      state['process_status'] = 'error_inconsistent_state'
                      return state
                      
                 poll_result = process.poll() 
                 if poll_result is None:
                     state["process_status"] = "running"
                     state["pid"] = process.pid
                 else:
                     # Process finished or terminated since last check
                     state["process_status"] = f"finished_with_code_{poll_result}"
                     state["pid"] = process.pid
                     logger.info(f"Detected training process {process.pid} for {project_id} finished with code {poll_result}.")
                     
                     # Determine final status and update state.json
                     # Cleanup is now handled by _update_project_state
                     current_project_status = state.get("current_step")
                     if current_project_status == "training" or current_project_status == "running": # Check if state needs update
                          final_status = "completed" if poll_result == 0 else "error"
                          error_info = None
                          if final_status == "error":
                              error_info = {"error": f"Training process exited with code {poll_result}. Check logs."}
                          # This call will also trigger cleanup
                          self._update_project_state(project_id, final_status, error_info) 
                          state["current_step"] = final_status # Reflect update in returned status
                          if error_info: state["status_details"] = error_info
                     else:
                          # State was already something else (e.g. cancelled, completed), just ensure cleanup happens
                           self._cleanup_training_artifacts(project_id)
                          
            elif state.get("current_step") == "training":
                 # Marked as training but no active process found
                 state["process_status"] = "unknown_or_orphaned"
                 # Attempt cleanup just in case temp dir was left behind
                 temp_dir_path = os.path.join(self.datasets_path, project_id, "images", "*_*") # Need pattern matching
                 # This is tricky, requires knowing repeats/class_name. Better to rely on active_processes dict.
                 # If it's not in active_processes, we assume cleanup happened or wasn't needed.
                 # Mark state as errored. Cleanup is implicitly handled by state update.
                 self._update_project_state(project_id, "error", {"error": "Training process not tracked (orphaned?)."})
                 state["current_step"] = "error" # Reflect update
                 state["status_details"] = {"error": "Training process not tracked (orphaned?)."}

            return state
        except Exception as e:
            logger.error(f"Error getting training status for {project_id}: {e}", exc_info=True)
            # Return last known state if possible, with an error indicator
            state["error_fetching_status"] = str(e)
            return state 

    def cancel_training(self, project_id: str) -> Dict[str, Any]:
        """Attempts to cancel an active training process."""
        logger.info(f"Received cancellation request for project {project_id}")
        
        process_entry = self.active_processes.get(project_id)
        
        if process_entry and process_entry.get("process"):
            process = process_entry["process"]
            poll_result = process.poll()
            if poll_result is None: # Process is still running
                try:
                    logger.warning(f"Terminating training process PID: {process.pid} for project {project_id}")
                    process.terminate() # Send SIGTERM first
                    try:
                        # Wait a short time for graceful termination
                        process.wait(timeout=5) 
                        logger.info(f"Process {process.pid} terminated gracefully.")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Process {process.pid} did not terminate gracefully, sending SIGKILL.")
                        process.kill() # Force kill if it doesn't terminate
                        process.wait() # Wait for kill to complete
                        logger.info(f"Process {process.pid} killed.")

                    # Update state (this will also trigger cleanup)
                    self._update_project_state(project_id, "cancelled", {"reason": "User initiated cancel"})
                    # No need to explicitly delete from active_processes here, _update_project_state calls _cleanup
                    return {"status": "cancelled", "message": "Training process terminated."}
                except Exception as e:
                    logger.error(f"Error terminating process {process.pid} for project {project_id}: {e}", exc_info=True)
                    # Ensure cleanup is attempted even on error
                    self._update_project_state(project_id, "error", {"error": f"Failed to terminate process: {str(e)}"})
                    raise HTTPException(status_code=500, detail=f"Error cancelling training: {str(e)}")
            else:
                # Process already finished - Cleanup handled by _update_project_state
                logger.info(f"Training process for {project_id} already finished with code {poll_result}. Cannot cancel.")
                if self._get_project_state_value(project_id, "current_step") == "training":
                     # ... (keep existing logic to update state for already finished process) ...
                     # Update state will trigger cleanup
                     self._update_project_state(project_id, final_status, error_info) 
                else:
                     # State is already non-training, ensure cleanup just in case
                     self._cleanup_training_artifacts(project_id) # Ensure cleanup happens if missed
                return {"status": "already_finished", "message": f"Training process already finished with code {poll_result}."}
        else:
            # No active process tracked
            # ... (keep existing logic for handling untracked processes) ...
            # Cleanup is implicitly handled by state update if needed
            if current_state == "training":
                self._update_project_state(project_id, "error", {"error": "Cancellation requested, but no active process was tracked."}) 
                return {"status": "error", "message": "No active training process found, state updated to error."}
            else:
                 return {"status": "not_running", "message": "No active training process found to cancel."}
                 
    def _get_project_state_value(self, project_id: str, key: str) -> Optional[Any]:
        """Helper to read a specific value from the project state file."""
        state_file = os.path.join(self.datasets_path, project_id, "metadata", "training_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    return state.get(key)
            except Exception as e:
                logger.warning(f"Could not read project state file {state_file}: {e}")
        return None

    def get_training_logs(self, project_id: str, log_type: str, lines: int = 100) -> str:
        """Retrieve the last N lines from the specified training log file."""
        if log_type not in ["stdout", "stderr"]:
            raise ValueError("Invalid log_type. Must be 'stdout' or 'stderr'.")

        log_dir = os.path.join(self.datasets_path, project_id, "log")
        log_filename = f"training_{log_type}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        logger.debug(f"Attempting to read log file: {log_filepath}")

        if not os.path.exists(log_filepath):
            logger.warning(f"Log file not found: {log_filepath}")
            # Raise specific error for API to handle as 404
            raise FileNotFoundError(f"Log file ({log_filename}) not found.")

        try:
            # Efficiently read last N lines (deque approach)
            from collections import deque
            with open(log_filepath, "r", encoding="utf-8") as f:
                last_lines = deque(f, maxlen=lines)
            log_content = "".join(last_lines)
                
            logger.debug(f"Successfully read {len(last_lines)} lines from {log_filepath}")
            return log_content
        except PermissionError as pe:
             logger.error(f"Permission denied reading log file {log_filepath}", exc_info=True)
             raise PermissionError(f"Permission denied reading log file: {log_filename}") # Raise specific error
        except Exception as e:
            logger.error(f"Error reading log file {log_filepath}: {e}", exc_info=True)
            # Raise a generic error for other issues
            raise RuntimeError(f"Error reading log file: {str(e)}") 