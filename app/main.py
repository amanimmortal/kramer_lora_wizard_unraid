from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import logging
from datetime import datetime

from .api import projects, training

# Configure logging
# log_dir = "logs" # Removed
# os.makedirs(log_dir, exist_ok=True) # Removed
# log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log") # Removed

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler(log_file), # Removed file logging
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Kramer LoRA Wizard")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Request: {request.method} {request.url}")
    logger.debug(f"Headers: {request.headers}")
    try:
        body = await request.body()
        if body:
            logger.debug(f"Body: {body.decode()}")
    except Exception as e:
        logger.debug(f"Could not read body: {e}")
    
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response

# Create data directories if they don't exist
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/datasets", exist_ok=True)

# Mount static files directory for data
app.mount("/data", StaticFiles(directory="data", html=False), name="data")

# Mount frontend static files
frontend_dist_path = "/app/frontend/dist"
index_path = os.path.join(frontend_dist_path, "index.html")
assets_path = os.path.join(frontend_dist_path, "assets")

# Include routers - separate prefixes to avoid conflicts
app.include_router(projects.router, prefix="/api/projects")
app.include_router(training.router, prefix="/api/training")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0"
    }

# 1. Serve index.html for the root path
@app.get("/")
async def serve_index():
    if not os.path.exists(index_path):
        logger.error(f"Frontend index.html not found at absolute path: {index_path}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Frontend index.html missing at {index_path}")
    return FileResponse(index_path)

# 2. Mount static assets directory (JS, CSS, etc.)
if os.path.exists(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
else:
     logger.warning(f"Frontend assets directory not found at absolute path: {assets_path}. Frontend may not load correctly.")

# 3. Mount static files directory for data access (/data/...)
# Placed after API routes to avoid conflicts if API used /data prefix
app.mount("/data", StaticFiles(directory="data", html=False), name="data") # html=False prevents serving index.html from data/

# 4. Catch-all route for client-side routing (React Router)
# This MUST be the LAST route defined
@app.get("/{full_path:path}")
async def serve_other_frontend_routes(full_path: str):
    # Exclude API, data, and asset paths explicitly
    if full_path.startswith("api/") or full_path.startswith("data/") or full_path.startswith("assets/") or full_path == "favicon.ico":
        # Let FastAPI handle potential 404s for actual missing API/data/asset routes
        # Or raise HTTPException if you want to be explicit
        logger.debug(f"Ignoring catch-all for non-frontend path: {full_path}")
        raise HTTPException(status_code=404, detail="Resource not found")
    
    logger.debug(f"Serving index.html for frontend route: {full_path}")
    if not os.path.exists(index_path):
        logger.error(f"Frontend index.html not found at absolute path: {index_path}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Frontend index.html missing at {index_path}")
    return FileResponse(index_path) 