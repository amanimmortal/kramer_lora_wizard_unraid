#!/bin/bash
set -e

# Ensure the logs directory is writable by appuser at runtime
# This handles potential volume mounts overwriting build-time permissions.
# echo "Ensuring /app/logs ownership..." # Removed
# sudo chown appuser:appgroup /app/logs # Removed

# Download models first
python download_models.py

# If downloads succeeded, start the main application
python run.py 