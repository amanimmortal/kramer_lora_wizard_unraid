# Build frontend
FROM node:18 AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Build backend
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
WORKDIR /app

# Add arguments for User and Group IDs
ARG PUID=1000
ARG PGID=1000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create group and user, allowing for existing GID/UID
RUN groupadd -f -g ${PGID} appgroup && \
    useradd -o -u ${PUID} -g ${PGID} -s /bin/bash -m appuser && \
    adduser appuser sudo

# Allow appuser (user name) to use sudo without password (more robust against UID/GID changes)
RUN echo 'appuser ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/appuser-nopasswd && \
    chmod 0440 /etc/sudoers.d/appuser-nopasswd

# Copy requirements file first for better Docker layer caching
COPY requirements.txt requirements.txt

# Install Python dependencies from requirements file
# Running as root here for system-wide installation
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BLIP model to avoid first-request delay
# Running as root here for system-wide cache or if it needs specific permissions
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')"

# --- Clone Kohya_ss sd-scripts and copy necessary files ---
# Running as root
RUN git clone https://github.com/kohya-ss/sd-scripts.git /tmp/sd-scripts
WORKDIR /tmp/sd-scripts
WORKDIR /app
# Copy required scripts and library folders
RUN cp /tmp/sd-scripts/train_network.py /app/
RUN cp /tmp/sd-scripts/sdxl_train_network.py /app/
RUN cp -r /tmp/sd-scripts/library /app/
RUN cp -r /tmp/sd-scripts/networks /app/
# Clean up
RUN rm -rf /tmp/sd-scripts
# --- End Kohya_ss sd-scripts ---

# Ensure the target app directory exists
RUN mkdir -p /app/app

# Copy the application components individually
COPY app/__init__.py /app/app/__init__.py
COPY app/main.py /app/app/main.py
COPY app/api/ /app/app/api/
COPY app/models/ /app/app/models/
COPY app/services/ /app/app/services/

# Copy other necessary files and directories
COPY data/templates/ /app/data/templates/
COPY run.py /app/run.py
COPY download_models.py /app/download_models.py
COPY start.sh /app/start.sh

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Create necessary data/log directories (still as root before chown)
RUN mkdir -p /app/data/models /app/data/datasets

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MKL_THREADING_LAYER=GNU

EXPOSE 8000

# Change ownership of the app directory and necessary subdirectories
RUN chown -R appuser:appgroup /app
RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Run startup script as appuser
CMD ["./start.sh"]