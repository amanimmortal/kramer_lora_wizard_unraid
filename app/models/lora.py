from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class LoraType(str, Enum):
    CHARACTER = "character"
    STYLE = "style"
    CONCEPT = "concept"

class ImageTag(BaseModel):
    tag: str
    confidence: Optional[float] = None

class ImageMetadata(BaseModel):
    id: str
    filename: str
    tags: List[str] = []
    caption: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None

class TrainingProgress(BaseModel):
    current_epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None

class TrainingSettings(BaseModel):
    base_model: str  # The selected base model (e.g. "IllustriousXL", "PonyXL", "SDXL")
    template_name: str  # The selected template name
    # Template overrides (optional)
    network_dim: Optional[int] = None
    network_alpha: Optional[int] = None
    resolution: Optional[str] = None
    learning_rate: Optional[float] = None
    unet_lr: Optional[float] = None
    text_encoder_lr: Optional[float] = None
    max_train_epochs: Optional[int] = None
    train_batch_size: Optional[int] = None
    save_every_n_epochs: Optional[int] = None
    mixed_precision: Optional[str] = None
    seed: Optional[int] = None
    noise_offset: Optional[float] = None
    keep_tokens: Optional[int] = None
    clip_skip: Optional[int] = None

class AutoTagSettings(BaseModel):
    label_type: str = "tag"  # "tag" or "caption"
    min_threshold: float = 0.35
    max_tags: int = 25
    blacklist_tags: List[str] = []
    prepend_tags: List[str] = []
    append_tags: List[str] = []
    existing_tags_mode: str = "append"  # "append", "overwrite", or "keep"

class ProjectState(BaseModel):
    current_step: str  # e.g. "created", "uploading", "tagging", "training", "completed"
    training_settings: Optional[TrainingSettings] = None
    training_progress: Optional[TrainingProgress] = None
    last_modified: str  # ISO format timestamp
    error: Optional[str] = None

class LoraProject(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    type: LoraType
    tags: List[str] = []
    images: List[ImageMetadata] = []
    triggerWord: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    last_modified: Optional[datetime] = Field(default_factory=datetime.utcnow)
    training_status: Optional[str] = None
    training_progress: Optional[float] = None
    training_settings: Optional[TrainingSettings] = None 