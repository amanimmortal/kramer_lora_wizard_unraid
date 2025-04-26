#!/bin/bash
set -e

# Download models first
python download_models.py

# If downloads succeeded, start the main application
python run.py 