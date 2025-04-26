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

# Copy the rest of the application
COPY app/ app/
COPY data/templates/ data/templates/
COPY run.py .
COPY download_models.py .
COPY start.sh .

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Create necessary directories (still as root before chown)
RUN mkdir -p data/models data/datasets logs

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MKL_THREADING_LAYER=GNU

EXPOSE 8000

# Change ownership of the app directory and necessary subdirectories
# This ensures the appuser can write to mounted volumes and install packages if needed later
RUN chown -R appuser:appgroup /app
RUN chmod +x start.sh

# Switch to non-root user
USER appuser

# Run startup script as appuser
CMD ["./start.sh"]