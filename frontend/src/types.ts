export enum LoraType {
    CHARACTER = "character",
    STYLE = "style",
    CONCEPT = "concept"
}

export interface ImageTag {
    tag: string;
    confidence?: number;
}

export interface ImageMetadata {
    id: string;
    filename: string;
    tags: string[];
    caption: string;
}

export interface TrainingProgress {
    current_epoch: number;
    total_epochs: number;
    loss: number;
    learning_rate: number;
}

export interface TrainingSettings {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    resolution: number;
    network_dim?: number;
    network_alpha?: number;
    clip_skip?: number;
    noise_offset?: number;
    keep_tokens?: number;
    min_bucket_reso?: number;
    max_bucket_reso?: number;
    shuffle_caption?: boolean;
    train_unet_only?: boolean;
    cache_latents?: boolean;
    cache_latents_to_disk?: boolean;
    optimizer_type?: string;
    mixed_precision?: string;
    enable_bucket?: boolean;
    bucket_reso_steps?: number;
    bucket_no_upscale?: boolean;
}

export interface LoraProject {
    id: string;
    name: string;
    type: LoraType;
    training_status: 'created' | 'uploading' | 'tagging' | 'training' | 'completed';
    created_at: string;
    last_modified?: string;
    images: ImageMetadata[];
    training_progress?: TrainingProgress;
    triggerWord?: string;
} 