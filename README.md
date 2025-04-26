# Kramer LoRA Wizard

A Docker-based tool for easy training of LoRA models.

## Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for training)
- At least 8GB of VRAM recommended

## Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/kramer_lora_wizard.git
   cd kramer_lora_wizard
   ```

2. Start the application:
   ```
   docker-compose up -d
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Development

All development is contained within Docker, no local installation of Node.js or Python required.

### Building Changes

If you make changes to the code, rebuild the Docker container:
```
docker-compose up --build -d
```

### Viewing Logs

```
docker-compose logs -f
```

### Stopping the Application

```
docker-compose down
```

## Project Structure

- `app/`: Python backend (FastAPI)
- `frontend/`: React frontend
- `data/`: Directory for images and trained models
  - `datasets/`: Contains uploaded image datasets
  - `models/`: Stores trained LoRA models
- `logs/`: Application logs

## Docker Volume Management

The application uses Docker volumes to persist data:
- `./data:/app/data`: Maps the local data directory to the container
- `./logs:/app/logs`: Maps logs directory
- `models:/app/data/models`: Volume for trained models

## Troubleshooting

- If the application fails to start, check the logs:
  ```
  docker-compose logs
  ```

- If you need to reset the application state:
  ```
  docker-compose down
  docker volume rm kramer_lora_wizard_models
  docker-compose up -d
  ```

- For GPU issues, verify Docker has access to your GPU:
  ```
  docker run --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
  ``` 