import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  IconButton,
  Stack,
  Paper,
  FormControl,
  RadioGroup,
  Radio,
  FormControlLabel,
  Slider,
  Divider,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  Add as AddIcon,
  ZoomIn as ZoomInIcon,
  ArrowForward as ArrowForwardIcon,
  Settings as SettingsIcon,
  Check as CheckIcon
} from '@mui/icons-material';
import axios from 'axios';
import { ImageMetadata } from '../types';
import { LoraProject } from '../types';

interface IndividualTagState {
  [imageId: string]: string;
}

interface AutoTagSettings {
  label_type: 'tag' | 'caption';
  existing_tags_mode: 'ignore' | 'append' | 'overwrite';
  max_tags: number;
  min_threshold: number;
  blacklist_tags: string[];
  prepend_tags: string[];
  append_tags: string[];
}

const TagImages: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [images, setImages] = useState<ImageMetadata[]>([]);
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [bulkNewTag, setBulkNewTag] = useState<string>('');
  const [individualTags, setIndividualTags] = useState<IndividualTagState>({});
  const [zoomedImage, setZoomedImage] = useState<ImageMetadata | null>(null);
  const [autoTagging, setAutoTagging] = useState(false);
  const [triggerWord, setTriggerWord] = useState<string>('');
  const [savingTriggerWord, setSavingTriggerWord] = useState(false);

  // --- State for the finalize step ---
  const [isFinalizing, setIsFinalizing] = useState<boolean>(false);
  const [finalizeError, setFinalizeError] = useState<string | null>(null);
  // --- End finalize state ---

  // Add auto-tag settings state
  const [autoTagSettingsOpen, setAutoTagSettingsOpen] = useState(false);
  const [autoTagSettings, setAutoTagSettings] = useState<AutoTagSettings>({
    label_type: 'tag',
    existing_tags_mode: 'ignore',
    max_tags: 10,
    min_threshold: 0.4,
    blacklist_tags: [],
    prepend_tags: [],
    append_tags: []
  });

  useEffect(() => {
    fetchProjectData();
  }, [projectId]);

  const fetchProjectData = async () => {
    if (!projectId) return;
    try {
      const response = await axios.get<LoraProject>(`/api/projects/${projectId}`);
      const projectData = response.data;
      setImages(projectData.images || []);
      setTriggerWord(projectData.triggerWord || '');
      const initialTags: IndividualTagState = {};
      (projectData.images || []).forEach(img => {
        initialTags[img.id] = '';
      });
      setIndividualTags(initialTags);
    } catch (error) {
      console.error('Error fetching project data:', error);
      setImages([]);
      setTriggerWord('');
    }
  };

  const handleSaveTriggerWord = async (newTriggerWord: string) => {
    if (!projectId) return;
    setSavingTriggerWord(true);
    try {
      await axios.post(`/api/projects/${projectId}/trigger-word`, { triggerWord: newTriggerWord });
      setTriggerWord(newTriggerWord);
    } catch (error) {
      console.error('Error saving trigger word:', error);
    } finally {
      setSavingTriggerWord(false);
    }
  };

  const handleAddTagsToImages = async (tagsToAdd: string[], imageIds: string[]) => {
    if (tagsToAdd.length === 0 || imageIds.length === 0) return;

    const tagPayload = tagsToAdd.map(tag => ({ tag: tag.trim() })).filter(t => t.tag);
    if (tagPayload.length === 0) return;

    try {
      const updates = imageIds.map((imageId) => {
        const image = images.find(img => img.id === imageId);
        if (!image) return Promise.resolve();

        return axios.post(`/api/projects/${projectId}/images/${image.filename}/tags`, tagPayload);
      });

      await Promise.all(updates.filter(p => p));

      await fetchProjectData();

      setBulkNewTag('');
      const updatedIndividualTags = { ...individualTags };
      imageIds.forEach(id => {
        if (updatedIndividualTags[id] !== undefined) {
          updatedIndividualTags[id] = '';
        }
      });
      setIndividualTags(updatedIndividualTags);

    } catch (error) {
      console.error('Error adding tag(s):', error);
    }
  };

  const handleTagRemove = async (imageId: string) => {
    try {
      const image = images.find((img) => img.id === imageId);
      if (!image || !image.tags) return;

      // NOTE: Current backend /tags endpoint MERGES tags. We need a way to *remove*.
      // This logic only refreshes the list for now.
      console.warn("Tag removal UI added, but backend needs update for true removal. Refreshing list.");
      await fetchProjectData();
    } catch (error) {
      console.error('Error removing tag:', error);
    }
  };

  const handleIndividualTagInputChange = (imageId: string, value: string) => {
    setIndividualTags(prev => ({ ...prev, [imageId]: value }));
  };

  const handleIndividualTagAdd = (imageId: string) => {
    const tag = individualTags[imageId]?.trim();
    if (tag) {
      handleAddTagsToImages([tag], [imageId]);
    }
  }

  const handleAutoTag = async () => {
    if (!projectId) return;
    setAutoTagging(true);
    setAutoTagSettingsOpen(false);
    try {
      await axios.post(`/api/training/${projectId}/auto-tag`, autoTagSettings);
      await fetchProjectData();  // Fetch images immediately after auto-tagging
    } catch (error) {
      console.error('Error auto-tagging:', error);
    } finally {
      setAutoTagging(false);
    }
  };

  const handleTagListChange = (
    field: 'blacklist_tags' | 'prepend_tags' | 'append_tags',
    value: string
  ) => {
    setAutoTagSettings({
      ...autoTagSettings,
      [field]: value.split(',').map(tag => tag.trim()).filter(Boolean)
    });
  };

  const handleProceed = async () => {
    if (!projectId) return;

    setIsFinalizing(true);
    setFinalizeError(null);

    try {
      // Call the new backend endpoint to ensure trigger word is in tags
      const response = await axios.post(`/api/projects/${projectId}/finalize-tags`);
      console.log('Finalize tags response:', response.data);
      // Navigate to training page on success
      navigate(`/train/${projectId}`);

    } catch (error: any) {
      console.error('Error finalizing tags:', error);
      const errorDetail = error.response?.data?.detail || 'Failed to ensure trigger word in tags.';
      setFinalizeError(errorDetail);
      // Don't navigate if there was an error
    } finally {
      setIsFinalizing(false);
    }
  }

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', mt: 4, p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Tag your images ({images.length} image{images.length !== 1 ? 's' : ''})
      </Typography>

      <TextField
        label="Trigger Word (prepended to tags)"
        value={triggerWord}
        onChange={(e) => setTriggerWord(e.target.value)}
        onBlur={(e) => handleSaveTriggerWord(e.target.value.trim())}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
        disabled={savingTriggerWord}
        InputProps={{
          endAdornment: savingTriggerWord ? <CircularProgress size={20} /> : null
        }}
      />

      <Paper sx={{ p: 2, mb: 4 }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center">
          <TextField
            size="small"
            label="Add tag to selected/all"
            value={bulkNewTag}
            onChange={(e) => setBulkNewTag(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && bulkNewTag) {
                const targetImageIds = selectedImages.length ? selectedImages : images.map(img => img.id);
                handleAddTagsToImages([bulkNewTag], targetImageIds);
              }
            }}
            sx={{ flexGrow: 1 }}
          />
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            disabled={!bulkNewTag}
            onClick={() => {
              const targetImageIds = selectedImages.length ? selectedImages : images.map(img => img.id);
              handleAddTagsToImages([bulkNewTag], targetImageIds);
            }}
          >
            Add to {selectedImages.length ? `${selectedImages.length} Selected` : 'All'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={() => setAutoTagSettingsOpen(true)}
            disabled={autoTagging}
          >
            Auto Tag Settings
          </Button>
          <Button
            variant="contained"
            color="primary"
            startIcon={isFinalizing ? <CircularProgress size={20} color="inherit" /> : <ArrowForwardIcon />}
            onClick={handleProceed}
            disabled={autoTagging || isFinalizing}
          >
            Proceed to Training
          </Button>
        </Stack>
        {selectedImages.length > 0 && (
          <Typography variant="caption" display="block" mt={1}>
            {selectedImages.length} image(s) selected. Click image to deselect.
          </Typography>
        )}
      </Paper>

      {finalizeError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {finalizeError}
        </Alert>
      )}

      <Grid container spacing={2}>
        {images.map((image) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={image.id}>
            <Card
              sx={{
                border: selectedImages.includes(image.id)
                  ? '2px solid'
                  : '1px solid',
                borderColor: selectedImages.includes(image.id) ? 'primary.main' : 'divider',
                height: '100%',
                display: 'flex',
                flexDirection: 'column'
              }}
            >
              <Box
                sx={{
                  position: 'relative',
                  aspectRatio: '1 / 1',
                  cursor: 'pointer',
                  overflow: 'hidden'
                }}
                onClick={() =>
                  setSelectedImages((prev) =>
                    prev.includes(image.id)
                      ? prev.filter((id) => id !== image.id)
                      : [...prev, image.id]
                  )
                }
              >
                <Box
                  component="img"
                  src={`/data/datasets/${projectId}/images/${image.filename}`}
                  alt={image.filename}
                  sx={{
                    display: 'block',
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                  }}
                />
                <IconButton
                  sx={{
                    position: 'absolute',
                    top: 4,
                    right: 4,
                    bgcolor: 'rgba(0,0,0,0.4)',
                    color: 'white',
                    '&:hover': {
                      bgcolor: 'rgba(0,0,0,0.6)'
                    }
                  }}
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    setZoomedImage(image);
                  }}
                >
                  <ZoomInIcon fontSize="small" />
                </IconButton>
              </Box>

              <CardContent sx={{ flexGrow: 1, pt: 1, pb: 0 }}>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
                  {triggerWord && (
                    <Chip
                      label={triggerWord}
                      size="small"
                      color="secondary"
                    />
                  )}
                  {(image.tags || []).map((tag) => (
                    tag !== triggerWord && (
                      <Chip
                        key={tag}
                        label={tag}
                        size="small"
                        onDelete={() => handleTagRemove(image.id)}
                      />
                    )
                  ))}
                </Box>
              </CardContent>

              <Box sx={{ p: 1, pt: 0.5, display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  size="small"
                  variant="outlined"
                  placeholder="Add tag..."
                  value={individualTags[image.id] || ''}
                  onChange={(e) => handleIndividualTagInputChange(image.id, e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleIndividualTagAdd(image.id);
                    }
                  }}
                />
                <IconButton size="small" onClick={() => handleIndividualTagAdd(image.id)} disabled={!individualTags[image.id]?.trim()}>
                  <AddIcon fontSize="small" />
                </IconButton>
              </Box>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog
        open={!!zoomedImage}
        onClose={() => setZoomedImage(null)}
        maxWidth="lg"
      >
        <DialogContent sx={{ p: 1 }}>
          {zoomedImage && (
            <Box
              component="img"
              src={`/data/datasets/${projectId}/images/${zoomedImage.filename}`}
              alt={zoomedImage.filename}
              sx={{
                display: 'block',
                maxWidth: '100%',
                maxHeight: '85vh',
                height: 'auto',
                objectFit: 'contain',
              }}
            />
          )}
        </DialogContent>
      </Dialog>

      {/* Auto-Tag Settings Dialog */}
      <Dialog
        open={autoTagSettingsOpen}
        onClose={() => setAutoTagSettingsOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Auto-Tagging Settings</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={3}>
            <FormControl fullWidth>
              <Typography gutterBottom>Label Type</Typography>
              <RadioGroup
                row
                value={autoTagSettings.label_type}
                onChange={(e) => setAutoTagSettings({
                  ...autoTagSettings,
                  label_type: e.target.value as 'tag' | 'caption'
                })}
              >
                <FormControlLabel
                  value="tag"
                  control={<Radio />}
                  label="Tags (multiple keywords)"
                />
                <FormControlLabel
                  value="caption"
                  control={<Radio />}
                  label="Caption (natural language)"
                />
              </RadioGroup>
            </FormControl>

            <FormControl fullWidth>
              <Typography gutterBottom>Existing Tags</Typography>
              <RadioGroup
                row
                value={autoTagSettings.existing_tags_mode}
                onChange={(e) => setAutoTagSettings({
                  ...autoTagSettings,
                  existing_tags_mode: e.target.value as 'ignore' | 'append' | 'overwrite'
                })}
              >
                <FormControlLabel
                  value="ignore"
                  control={<Radio />}
                  label="Ignore"
                />
                <FormControlLabel
                  value="append"
                  control={<Radio />}
                  label="Append"
                />
                <FormControlLabel
                  value="overwrite"
                  control={<Radio />}
                  label="Overwrite"
                />
              </RadioGroup>
            </FormControl>

            <FormControl fullWidth>
              <Typography gutterBottom>
                Max Tags: {autoTagSettings.max_tags}
              </Typography>
              <Slider
                value={autoTagSettings.max_tags}
                onChange={(_, value) => setAutoTagSettings({
                  ...autoTagSettings,
                  max_tags: value as number
                })}
                min={1}
                max={50}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 10, label: '10' },
                  { value: 25, label: '25' },
                  { value: 50, label: '50' }
                ]}
                valueLabelDisplay="auto"
              />
            </FormControl>

            <FormControl fullWidth>
              <Typography gutterBottom>
                Minimum Confidence Threshold: {autoTagSettings.min_threshold.toFixed(1)}
              </Typography>
              <Slider
                value={autoTagSettings.min_threshold}
                onChange={(_, value) => setAutoTagSettings({
                  ...autoTagSettings,
                  min_threshold: value as number
                })}
                min={0}
                max={1}
                step={0.1}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' }
                ]}
                valueLabelDisplay="auto"
              />
            </FormControl>

            <Divider />

            <FormControl fullWidth>
              <TextField
                label="Tag Blacklist (comma-separated)"
                value={autoTagSettings.blacklist_tags.join(', ')}
                onChange={(e) => handleTagListChange('blacklist_tags', e.target.value)}
                multiline
                rows={2}
                placeholder="e.g., nsfw, bad, text"
                variant="outlined"
                helperText="Tags to exclude from results"
              />
            </FormControl>

            <FormControl fullWidth>
              <TextField
                label="Prepend Tags (comma-separated)"
                value={autoTagSettings.prepend_tags.join(', ')}
                onChange={(e) => handleTagListChange('prepend_tags', e.target.value)}
                multiline
                rows={2}
                placeholder="e.g., masterpiece, best quality"
                variant="outlined"
                helperText="Tags that will be added before auto-generated tags"
              />
            </FormControl>

            <FormControl fullWidth>
              <TextField
                label="Append Tags (comma-separated)"
                value={autoTagSettings.append_tags.join(', ')}
                onChange={(e) => handleTagListChange('append_tags', e.target.value)}
                multiline
                rows={2}
                placeholder="e.g., high resolution, detailed"
                variant="outlined"
                helperText="Tags that will be added after auto-generated tags"
              />
            </FormControl>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAutoTagSettingsOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="contained"
            color="primary"
            onClick={handleAutoTag}
            disabled={autoTagging}
          >
            {autoTagging ? 'Auto-tagging...' : 'Auto-Tag All Images'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TagImages; 