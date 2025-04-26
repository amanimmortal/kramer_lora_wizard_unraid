#!/bin/bash
set -e

# Ensure critical directories are accessible/writable by appuser at runtime
# This handles potential volume mounts overwriting build-time permissions.
echo "Ensuring /app/data/templates permissions..."
sudo chown -R appuser:appgroup /app/data/templates
sudo chmod -R u+r /app/data/templates # Read-only needed

echo "Ensuring /app/data/datasets permissions..."
sudo chown -R appuser:appgroup /app/data/datasets
sudo chmod -R u+rw /app/data/datasets # Read/Write needed

echo "Ensuring /app/data/models permissions..."
sudo chown -R appuser:appgroup /app/data/models
sudo chmod -R u+rw /app/data/models # Read/Write needed

# Download models first
python download_models.py

# If downloads succeeded, start the main application
python run.py 