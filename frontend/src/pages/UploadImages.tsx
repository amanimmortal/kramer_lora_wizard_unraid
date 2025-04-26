/// <reference types="react" />
/// <reference types="react-router-dom" />
/// <reference types="react-dropzone" />
/// <reference types="jszip" />
/// <reference types="@mui/material" />

import React, { useCallback, useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import JSZip from 'jszip';
import {
  Box,
  Typography,
  Button,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Paper,
  Stack
} from '@mui/material';

interface FileWithPreview extends File {
  preview?: string;
}

interface UploadState {
  uploading: boolean;
  progress: number;
  error: string | null;
  files: FileWithPreview[];
  existingImageCount: number;
}

const UploadImages: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [state, setState] = useState<UploadState>({
    uploading: false,
    progress: 0,
    error: null,
    files: [],
    existingImageCount: 0
  });

  useEffect(() => {
    const fetchProjectData = async () => {
      if (!projectId) return;
      try {
        const response = await fetch(`/api/projects/${projectId}`);
        if (response.ok) {
          const projectData = await response.json();
          setState(prev => ({ ...prev, existingImageCount: projectData.images?.length ?? 0 }));
        } else {
          console.error('Failed to fetch project data', response.statusText);
        }
      } catch (error) {
        console.error('Error fetching project data:', error);
      }
    };
    fetchProjectData();
  }, [projectId]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setState((prev: UploadState) => ({
      ...prev,
      error: null,
      files: [],
      uploading: false,
      progress: 0
    }));

    const processedFiles: FileWithPreview[] = [];
    const errors: string[] = [];

    for (const file of acceptedFiles) {
      if (file.name.toLowerCase().endsWith('.zip')) {
        try {
          const zip = new JSZip();
          const contents = await zip.loadAsync(file);

          await Promise.all(
            Object.values(contents.files).map(async (zipEntry: JSZip.JSZipObject) => {
              if (!zipEntry.dir && /\.(jpg|jpeg|png|webp)$/i.test(zipEntry.name)) {
                const blob = await zipEntry.async('blob');
                const imageFile = new File([blob], zipEntry.name, { type: `image/${zipEntry.name.split('.').pop()}` });
                processedFiles.push(Object.assign(imageFile, {
                  preview: URL.createObjectURL(blob)
                }));
              }
            })
          );
        } catch (error) {
          errors.push(`Failed to process ZIP file ${file.name}: ${error}`);
        }
      } else if (/\.(jpg|jpeg|png|webp)$/i.test(file.name)) {
        processedFiles.push(Object.assign(file, {
          preview: URL.createObjectURL(file)
        }));
      } else {
        errors.push(`Unsupported file type: ${file.name}`);
      }
    }

    if (errors.length > 0) {
      setState((prev: UploadState) => ({ ...prev, error: errors.join('\n') }));
    }

    setState((prev: UploadState) => ({ ...prev, files: [...prev.files, ...processedFiles] }));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.webp'],
      'application/zip': ['.zip']
    }
  });

  const uploadFiles = async () => {
    if (state.files.length === 0) return;

    setState((prev: UploadState) => ({ ...prev, uploading: true, progress: 0 }));
    const totalFiles = state.files.length;
    let uploadedFiles = 0;

    try {
      for (const file of state.files) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`/api/projects/${projectId}/images`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`Failed to upload ${file.name}`);
        }

        uploadedFiles++;
        setState((prev: UploadState) => ({
          ...prev,
          progress: (uploadedFiles / totalFiles) * 100
        }));
      }

      state.files.forEach((file: FileWithPreview) => {
        if (file.preview) {
          URL.revokeObjectURL(file.preview);
        }
      });

      if (projectId) {
        navigate(`/tag/${projectId}`);
      }

      setState((prev: UploadState) => ({
        ...prev,
        uploading: false,
        files: [],
        progress: 100,
        existingImageCount: prev.existingImageCount + totalFiles
      }));

    } catch (error) {
      setState((prev: UploadState) => ({
        ...prev,
        uploading: false,
        error: error instanceof Error ? error.message : 'Upload failed'
      }));
    }
  };

  const handleProceedToTagging = () => {
    if (projectId) {
      navigate(`/tag/${projectId}`);
    }
  }

  return (
    <Box p={4} maxWidth={1200} mx="auto">
      <Typography variant="h4" mb={4}>Upload Images</Typography>

      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          mb: 4,
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
          cursor: 'pointer'
        }}
      >
        <input {...getInputProps()} />
        <Typography align="center">
          {isDragActive
            ? 'Drop the files here...'
            : 'Drag and drop image files or ZIP archives here, or click to select files'
          }
        </Typography>
      </Paper>

      {state.error && (
        <Typography color="error" mb={2}>
          {state.error}
        </Typography>
      )}

      {state.existingImageCount > 0 && !state.files.length && (
        <Typography variant="body1" mb={2}>
          Project already contains {state.existingImageCount} image(s).
        </Typography>
      )}

      {state.files.length > 0 && (
        <>
          <List>
            {state.files.map((file: FileWithPreview, index: number) => (
              <ListItem key={index}>
                <ListItemText
                  primary={file.name}
                  secondary={`${(file.size / 1024 / 1024).toFixed(2)} MB`}
                />
                {file.preview && (
                  <Box
                    component="img"
                    src={file.preview}
                    sx={{
                      width: 50,
                      height: 50,
                      objectFit: 'cover',
                      borderRadius: 1
                    }}
                  />
                )}
              </ListItem>
            ))}
          </List>
        </>
      )}

      <Stack direction="row" spacing={2} mt={2}>
        {state.files.length > 0 && (
          <Button
            variant="contained"
            onClick={uploadFiles}
            disabled={state.uploading}
          >
            Upload {state.files.length} file(s)
          </Button>
        )}

        {state.existingImageCount > 0 && state.files.length === 0 && (
          <Button
            variant="contained"
            onClick={handleProceedToTagging}
            disabled={state.uploading}
          >
            Proceed to Tagging ({state.existingImageCount} images)
          </Button>
        )}
      </Stack>

      {state.uploading && (
        <Box mt={2}>
          <LinearProgress variant="determinate" value={state.progress} />
          <Typography variant="body2" color="text.secondary" mt={1}>
            Uploading... {Math.round(state.progress)}%
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default UploadImages; 