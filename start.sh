#!/bin/bash
set -e

echo "Listing /app contents:"
ls -la /app
echo "--------------------"
echo "Listing /app/app contents:"
ls -la /app/app
echo "--------------------"
echo "Listing /app/app/models contents:"
ls -la /app/app/models
echo "--------------------"

# Download models first
python download_models.py

# If downloads succeeded, start the main application
python run.py 