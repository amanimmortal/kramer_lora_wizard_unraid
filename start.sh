#!/bin/bash
set -e

# Ensure critical directories are accessible/writable by appuser at runtime
# This handles potential volume mounts overwriting build-time permissions.
echo "Ensuring /app/data/templates permissions..."
# sudo chown -R appuser:appgroup /app/data/templates # Removed - handled by Dockerfile chown + PUID/PGID
# sudo chmod -R u+r /app/data/templates # Removed - should inherit from chown

echo "Ensuring /app/data/datasets permissions..."
# sudo chown -R appuser:appgroup /app/data/datasets # Removed - handled by Dockerfile chown + PUID/PGID
# sudo chmod -R u+rw /app/data/datasets # Removed - should inherit from chown

echo "Ensuring /app/data/models permissions..."
# sudo chown -R appuser:appgroup /app/data/models # Removed - handled by Dockerfile chown + PUID/PGID
# sudo chmod -R u+rw /app/data/models # Removed - should inherit from chown

# Download models first
python download_models.py

# If downloads succeeded, start the main application
python run.py 