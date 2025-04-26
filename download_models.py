import os
import sys
import time
import requests
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder

MODELS = {
    'sdxl': {
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors',
        'filename': 'sdxl.safetensors',
        'requires_auth': False
    },
    'ponyxl': {
        'url': 'https://huggingface.co/Magamanny/Pony-Diffusion-V6-XL/resolve/main/ponyDiffusionV6XL_v6StartWithThisOne.safetensors?download=true',
        'filename': 'ponyDiffusionV6XL_v6StartWithThisOne.safetensors',
        'requires_auth': False
    },
    'illustriousxl': {
        'url': 'https://huggingface.co/Liberata/illustrious-xl-v1.0/resolve/main/Illustrious-XL-v1.0.safetensors?download=true',
        'filename': 'Illustrious-XL-v1.0.safetensors',
        'requires_auth': False
    }
}

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
SIZE_MISMATCH_TOLERANCE = 1024 * 1024  # 1MB tolerance for size differences

def get_auth_header():
    token = os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        print("HUGGINGFACE_TOKEN environment variable not set. Some models require authentication.")
        print("Please set your Hugging Face token in the environment:")
        print("  1. Get your token from https://huggingface.co/settings/tokens")
        print("  2. Add it to your docker-compose.yml environment section")
        return None
    return {"Authorization": f"Bearer {token}"}

def verify_file_size(filepath: Path, expected_size: int) -> bool:
    """Verify if file exists and has the expected size within tolerance."""
    if not filepath.exists():
        return False
    
    actual_size = filepath.stat().st_size
    size_difference = abs(actual_size - expected_size)
    
    if size_difference > SIZE_MISMATCH_TOLERANCE:
        print(f"File size mismatch for {filepath.name}")
        print(f"Expected: {expected_size:,} bytes")
        print(f"Actual: {actual_size:,} bytes")
        print(f"Difference: {size_difference:,} bytes (tolerance: {SIZE_MISMATCH_TOLERANCE:,} bytes)")
        print("Deleting file and retrying download...")
        filepath.unlink()
        return False
    return True

def get_file_size(url: str, headers: dict = None) -> int:
    """Get expected file size from Content-Length header."""
    try:
        response = requests.head(url, headers=headers, allow_redirects=True)
        if response.ok:
            content_length = response.headers.get('content-length')
            if content_length:
                return int(content_length)
            print(f"Warning: No Content-Length header for {url}")
    except Exception as e:
        print(f"Failed to get file size: {e}")
    return 0

def download_with_progress(url: str, filepath: Path, headers: dict = None) -> tuple[bool, int]:
    """Download file with progress bar and handle interruptions. Returns (success, actual_size)."""
    temp_filepath = filepath.with_suffix(filepath.suffix + '.temp')
    
    try:
        response = requests.get(url, stream=True, headers=headers)
        if not response.ok:
            print(f"Failed to download: {response.status_code}")
            if response.status_code == 401:
                print("Authentication failed. Please check your Hugging Face token.")
            return False, 0

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunks
        downloaded_size = 0

        with open(temp_filepath, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    data_len = len(data)
                    downloaded_size += data_len
                    pbar.update(data_len)

        if downloaded_size == total_size or total_size == 0:
            temp_filepath.rename(filepath)
            return True, downloaded_size
        else:
            print(f"Download size mismatch. Expected: {total_size:,}, Got: {downloaded_size:,}")
            if temp_filepath.exists():
                temp_filepath.unlink()
            return False, 0

    except Exception as e:
        print(f"Download error: {e}")
        if temp_filepath.exists():
            temp_filepath.unlink()
        return False, 0

def download_file(model_name: str, model_info: dict, models_dir: Path) -> bool:
    """Download file with retries and integrity checks."""
    filepath = models_dir / model_info['filename']
    
    # Get the URL for the model
    if model_info.get('requires_auth', False):
        headers = get_auth_header()
        if not headers:
            print(f"Skipping {model_name} - authentication required")
            return False
        url = f"https://huggingface.co/{model_info['repo_id']}/resolve/main/{model_info['filename']}"
    else:
        headers = None
        url = model_info['url']

    # Get expected file size from server
    expected_size = get_file_size(url, headers)
    if expected_size == 0:
        print(f"Warning: Could not determine expected size for {model_info['filename']}")
    
    # Check if existing file is complete
    if filepath.exists() and expected_size > 0:
        if verify_file_size(filepath, expected_size):
            print(f"Model {model_info['filename']} already exists and is complete, skipping download...")
            return True

    print(f"Downloading {model_info['filename']}...")
    
    # Try downloading with retries
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            print(f"Retry attempt {attempt + 1} of {MAX_RETRIES}")
            time.sleep(RETRY_DELAY)

        success, downloaded_size = download_with_progress(url, filepath, headers)
        if success:
            if expected_size == 0 or verify_file_size(filepath, downloaded_size):
                return True
            print("File verification failed after download")
        
        if filepath.exists():
            filepath.unlink()

    print(f"Failed to download {model_name} after {MAX_RETRIES} attempts")
    return False

def main():
    models_dir = Path("/app/data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for model_name, model_info in MODELS.items():
        if not download_file(model_name, model_info, models_dir):
            success = False
            print(f"Failed to download {model_name}")

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 