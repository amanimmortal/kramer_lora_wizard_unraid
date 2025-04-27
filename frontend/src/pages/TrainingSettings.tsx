import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Checkbox,
  FormControlLabel,
  Paper,
  Chip,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import axios from 'axios';
import { LoraProject } from '../types';

// Define type for loaded settings (can be more specific based on JSON structure)
type TemplateSettings = any;

const TrainingSettings: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [project, setProject] = useState<LoraProject | null>(null);
  const [baseModel, setBaseModel] = useState<string>('');
  const [settings, setSettings] = useState<TemplateSettings | null>(null);
  const [availableBaseModels, setAvailableBaseModels] = useState<string[]>([]);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingProject, setLoadingProject] = useState<boolean>(true);
  const [loadingTemplate, setLoadingTemplate] = useState<boolean>(false);
  const [repeats, setRepeats] = useState<number>(10);

  // --- New state for status polling and logs ---
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null); // e.g., 'training', 'completed', 'error', 'running'
  const [logContent, setLogContent] = useState<string>('');
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null); // Ref to store interval ID
  const logContainerRef = useRef<HTMLDivElement>(null); // Ref for auto-scrolling logs
  // --- End new state ---

  // Effect to fetch available models
  useEffect(() => {
    const fetchAvailableModels = async () => {
      try {
        const response = await axios.get('/api/training/base-models');
        setAvailableBaseModels(response.data.models);
      } catch (error) {
        console.error('Error fetching available models:', error);
        setError('Failed to load available models.');
      }
    };
    fetchAvailableModels();
  }, []); // Runs once on mount

  // Effect to fetch project details
  useEffect(() => {
    const fetchProjectDetails = async () => {
      if (!projectId) return;
      setLoadingProject(true);
      try {
        const response = await axios.get(`/api/projects/${projectId}`);
        setProject(response.data);

        // Calculate repeats based on project type and number of images
        const numImages = response.data.images?.length || 0;
        const epochs = settings?.max_train_epochs || 10; // Default to 10 if not set
        const targetSteps = response.data.type === 'character' ? 2000 : 5000;

        // Calculate repeats to achieve target steps
        const calculatedRepeats = Math.max(1, Math.round(targetSteps / (numImages * epochs)));
        setRepeats(calculatedRepeats);
      } catch (error) {
        console.error('Error fetching project details:', error);
        setError('Failed to load project details.');
      } finally {
        setLoadingProject(false);
      }
    };
    fetchProjectDetails();
  }, [projectId, settings?.max_train_epochs]);

  // Effect to load template settings when base model changes
  useEffect(() => {
    const loadTemplateSettings = async () => {
      if (!baseModel || !project?.type) return; // Ensure project type is also available

      setLoadingTemplate(true);
      setError(null);
      setSettings(null); // Clear old settings

      // --- Mapping from full model name to short template key ---
      const modelNameToShortKey = (fullModelName: string): string => {
        const nameLower = fullModelName.toLowerCase();
        if (nameLower.includes('illustrious')) return 'illustriousxl';
        if (nameLower.includes('pony')) return 'ponyxl';
        if (nameLower.includes('sdxl')) return 'sdxl';
        // Add more mappings if needed
        console.warn(`No short key mapping found for model: ${fullModelName}, using lowercase.`);
        return nameLower; // Fallback, might fail
      };
      // --- End Mapping ---

      try {
        // Fetch ALL templates first
        const response = await axios.get(`/api/training/templates`);
        // Correctly access the templates object from the response data
        const templates = response.data;

        // Use the mapping function to get the short key
        const shortModelKey = modelNameToShortKey(baseModel);
        const templateKey = `${shortModelKey}_${project.type === 'character' ? 'char' : 'style'}`;

        console.debug(`Attempting to load template with key: ${templateKey} for base model ${baseModel}`);

        if (templates[templateKey]) {
          setSettings(templates[templateKey]);
        } else {
          console.error(`Template key '${templateKey}' not found in available templates:`, Object.keys(templates));
          setError(`No template found for ${baseModel} (${project.type}). Key '${templateKey}' missing.`);
        }
      } catch (error) {
        console.error('Error loading template settings:', error);
        setError('Failed to load template settings.');
      } finally {
        setLoadingTemplate(false);
      }
    };

    loadTemplateSettings();
  }, [baseModel, project?.type]); // Depend on project.type as well

  // Function to fetch status and logs
  const fetchStatusAndLogs = useCallback(async () => {
    if (!projectId) return;
    let currentStdout = '';
    let currentStderr = '';
    let finalStatus = trainingStatus; // Keep track of status within the fetch

    try {
      // Fetch Status
      const statusRes = await axios.get(`/api/training/${projectId}/training-status`);
      finalStatus = statusRes.data?.current_step || statusRes.data?.process_status || 'unknown';
      setTrainingStatus(finalStatus);

      const isPotentiallyRunning = finalStatus === 'training' || finalStatus === 'running';

      // Fetch Stdout Logs
      try {
        const stdoutRes = await axios.get(`/api/training/${projectId}/logs?log_type=stdout&lines=200`);
        currentStdout = stdoutRes.data;
      } catch (logErr) {
        console.warn("Error fetching stdout logs:", logErr);
        currentStdout = "Error fetching stdout logs.";
      }

      // Fetch Stderr Logs (always fetch if potentially running or if finished/errored on this poll)
      if (isPotentiallyRunning || finalStatus === 'completed' || finalStatus === 'error') {
        try {
          const stderrRes = await axios.get(`/api/training/${projectId}/logs?log_type=stderr&lines=200`);
          currentStderr = stderrRes.data;
        } catch (logErr) {
          console.warn("Error fetching stderr logs:", logErr);
          currentStderr = "Error fetching stderr logs.";
        }
      }

      // Combine logs
      let combinedLogs = currentStdout;
      if (currentStderr && currentStderr.trim() !== "Log file (training_stderr.log) not found.") {
        if (combinedLogs) combinedLogs += "\n\n--- STDERR ---\n";
        combinedLogs += currentStderr;
      }
      setLogContent(combinedLogs || (isPolling ? 'Waiting for logs...' : 'No logs to display.'));

      // Auto-scroll logs
      if (logContainerRef.current) {
        logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
      }

      // Stop polling check
      if (!isPotentiallyRunning) {
        setIsPolling(false); // Stop polling if completed or error
      }

    } catch (err) {
      console.error("Error fetching status:", err);
      // Stop polling on status fetch error
      setIsPolling(false);
      setError("Error fetching training status.");
      setTrainingStatus("error");
    }
  }, [projectId, isPolling]); // Removed trainingStatus dependency to prevent potential loop issues

  // Effect to manage polling interval
  useEffect(() => {
    if (isPolling && projectId) {
      // Start polling
      fetchStatusAndLogs(); // Fetch immediately first time
      pollIntervalRef.current = setInterval(fetchStatusAndLogs, 5000); // Poll every 5 seconds
      console.log("Polling started");
    } else {
      // Stop polling
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
        console.log("Polling stopped");
      }
    }
    // Cleanup function to stop polling when component unmounts
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        console.log("Polling stopped on unmount");
      }
    };
  }, [isPolling, projectId, fetchStatusAndLogs]);

  // Helper function to update a setting value in the state
  // Handles nested updates correctly
  const updateSetting = useCallback((key: string, value: any) => {
    setSettings((prevSettings) => {
      if (!prevSettings) return null;

      // Special handling for resolution: update the string format
      let valueToSet = value;
      if (key === 'resolution') {
        const parsedValue = parseInt(value, 10);
        if (!isNaN(parsedValue)) {
          valueToSet = `${parsedValue},${parsedValue}`; // Assuming square aspect ratio update
        } else {
          valueToSet = prevSettings.resolution; // Keep original if invalid input
        }
      }

      // Create a deep copy to avoid direct state mutation
      const newSettings = JSON.parse(JSON.stringify(prevSettings));
      newSettings[key] = valueToSet;
      return newSettings;
    });
  }, []);

  // --- Helper function to render the correct input type ---
  const renderSettingInput = (key: string, value: any) => {
    const label = key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    const commonProps = { key, label, fullWidth: true, variant: "outlined" as const, size: "small" as const };

    // Boolean -> Checkbox
    if (typeof value === 'boolean') {
      return (
        <FormControlLabel
          control={
            <Checkbox
              checked={value}
              onChange={(e) => updateSetting(key, e.target.checked)}
            />
          }
          label={label}
          sx={{ width: '100%', justifyContent: 'space-between', ml: 0, mr: 0.5 }} // Style checkbox nicely
          labelPlacement="start"
        />
      );
    }

    // Number -> Number TextField (or Slider for known ranges)
    if (typeof value === 'number') {
      // Basic number input
      return (
        <TextField
          {...commonProps}
          type="number"
          value={value}
          onChange={(e) => {
            const num = key === 'learning_rate' || key === 'unet_lr' || key === 'text_encoder_lr' || key === 'noise_offset' || key === 'min_snr_gamma' || key === 'network_dropout' || key === 'scale_weight_norms'
              ? parseFloat(e.target.value)
              : parseInt(e.target.value, 10);
            updateSetting(key, isNaN(num) ? value : num); // Keep old value if parse fails
          }}
          // Add step for float values if desired
          inputProps={{
            step: key === 'learning_rate' || key.endsWith('_lr') || key === 'noise_offset' || key === 'min_snr_gamma' || key === 'network_dropout' || key === 'scale_weight_norms' ? '0.00001' : '1'
          }}
        />
      );
    }

    // String -> TextField (potentially multiline or Select)
    if (typeof value === 'string') {
      // Example: Could use Select for known options like optimizer_type, lr_scheduler etc.
      if (key === 'optimizer_type') {
        return (
          <FormControl fullWidth size="small">
            <InputLabel>{label}</InputLabel>
            <Select value={value} label={label} onChange={(e) => updateSetting(key, e.target.value)}>
              {/* Add known optimizer types */}
              <MenuItem value="AdamW">AdamW</MenuItem>
              <MenuItem value="AdamW8bit">AdamW8bit</MenuItem>
              <MenuItem value="Prodigy">Prodigy</MenuItem>
              <MenuItem value="Adafactor">Adafactor</MenuItem>
              <MenuItem value="DAdaptation">DAdaptation</MenuItem>
              {/* Add others if needed */}
              {!["AdamW", "AdamW8bit", "Prodigy", "Adafactor", "DAdaptation"].includes(value) && (
                <MenuItem value={value}><em>{value} (Custom)</em></MenuItem>
              )}
            </Select>
          </FormControl>
        );
      }
      if (key === 'lr_scheduler') {
        return (
          <FormControl fullWidth size="small">
            <InputLabel>{label}</InputLabel>
            <Select value={value} label={label} onChange={(e) => updateSetting(key, e.target.value)}>
              <MenuItem value="constant">constant</MenuItem>
              <MenuItem value="constant_with_warmup">constant_with_warmup</MenuItem>
              <MenuItem value="linear">linear</MenuItem>
              <MenuItem value="cosine">cosine</MenuItem>
              <MenuItem value="cosine_with_restarts">cosine_with_restarts</MenuItem>
              <MenuItem value="polynomial">polynomial</MenuItem>
              {!["constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"].includes(value) && (
                <MenuItem value={value}><em>{value} (Custom)</em></MenuItem>
              )}
            </Select>
          </FormControl>
        );
      }
      if (key === 'mixed_precision' || key === 'save_precision') {
        return (
          <FormControl fullWidth size="small">
            <InputLabel>{label}</InputLabel>
            <Select value={value} label={label} onChange={(e) => updateSetting(key, e.target.value)}>
              <MenuItem value="no">no</MenuItem>
              <MenuItem value="fp16">fp16</MenuItem>
              <MenuItem value="bf16">bf16</MenuItem>
              {!["no", "fp16", "bf16"].includes(value) && (
                <MenuItem value={value}><em>{value} (Custom)</em></MenuItem>
              )}
            </Select>
          </FormControl>
        );
      }
      if (key === 'save_model_as') {
        return (
          <FormControl fullWidth size="small">
            <InputLabel>{label}</InputLabel>
            <Select value={value} label={label} onChange={(e) => updateSetting(key, e.target.value)}>
              <MenuItem value="safetensors">safetensors</MenuItem>
              <MenuItem value="ckpt">ckpt</MenuItem>
            </Select>
          </FormControl>
        );
      }

      // Default to TextField
      return (
        <TextField
          {...commonProps}
          value={value}
          onChange={(e) => updateSetting(key, e.target.value)}
          multiline={key === 'optimizer_args' || key === 'sample_prompts'} // Example: Multiline for certain fields
          rows={key === 'optimizer_args' || key === 'sample_prompts' ? 3 : 1}
        />
      );
    }

    // Handle null or other types (display as read-only or skip)
    return (
      <TextField
        {...commonProps}
        value={value === null ? 'null' : JSON.stringify(value)}
        disabled // Non-editable for unsupported types
        InputProps={{ readOnly: true }}
      />
    );
  };

  // Add this mapping function above handleStartTraining
  const filenameToShortKey = (filename: string): string => {
    if (filename === 'Illustrious-XL-v1.0') return 'illustriousxl';
    if (filename === 'ponyDiffusionV6XL_v6StartWithThisOne') return 'ponyxl';
    if (filename === 'sdxl') return 'sdxl';
    return filename.toLowerCase(); // fallback
  };

  const handleStartTraining = async () => {
    if (!settings || !baseModel || !project || !repeats || baseModel === "undefined") {
      setError("Please select a valid base model before starting training.");
      return;
    }
    setTraining(true); // Indicate API call in progress
    setError(null);
    setLogContent(''); // Clear previous logs
    setTrainingStatus('starting'); // Set initial status

    // Process optimizer_args if present - convert comma-separated to space-separated
    let processedSettings = { ...settings };
    if (processedSettings.optimizer_args && typeof processedSettings.optimizer_args === 'string') {
      // Replace commas with spaces for proper parsing
      processedSettings.optimizer_args = processedSettings.optimizer_args.replace(/,/g, ' ');
    }

    const shortKey = filenameToShortKey(baseModel);

    const payload = {
      baseModelName: shortKey, // use short key for backend
      repeats: repeats,
      settings: {
        ...processedSettings,
        pretrained_model_name_or_path: `/app/data/models/${baseModel}.safetensors`, // use filename for path
        train_data_dir: `/app/data/datasets/${projectId}/images`,
        output_dir: `/app/data/datasets/${projectId}/output`,
        logging_dir: `/app/data/datasets/${projectId}/log`,
        output_name: `${project.name}_${baseModel}`,
      }
    };
    console.log("baseModel:", baseModel, "shortKey:", shortKey, "payload:", payload);

    try {
      await axios.post(`/api/training/${projectId}/train`, payload);
      setTrainingStatus('training'); // Update status
      setIsPolling(true); // Start polling for status and logs
    } catch (error: any) {
      console.error('Error starting training:', error);
      const errorDetail = error.response?.data?.detail || 'Failed to start training';
      setError(errorDetail);
      setTrainingStatus('error'); // Set status to error
      setLogContent(`Error starting training: ${errorDetail}`); // Show error in log area
      setIsPolling(false); // Ensure polling is off
    } finally {
      setTraining(false); // API call finished
    }
  };

  const handleCancelTraining = async () => {
    if (!projectId) return;
    const confirmCancel = window.confirm("Are you sure you want to cancel the current training run?");
    if (!confirmCancel) return;

    console.log(`Attempting to cancel training for project ${projectId}`);
    setTraining(true); // Use the training state to disable buttons during API call
    setError(null); // Clear previous errors

    try {
      const response = await axios.post(`/api/training/${projectId}/cancel`);
      console.log("Cancel response:", response.data);
      // Update status based on response
      setTrainingStatus(response.data?.status || 'cancelled');
      setIsPolling(false); // Stop polling immediately after cancel request
      setLogContent((prev) => prev + "\n\n--- Cancellation Requested ---");
      alert("Training cancellation requested."); // Optional user feedback
    } catch (error: any) {
      console.error('Error cancelling training:', error);
      const errorDetail = error.response?.data?.detail || 'Failed to cancel training';
      setError(errorDetail);
      // Keep polling maybe? Or set status to error? For now, just show error.
      // setIsPolling(false); 
    } finally {
      setTraining(false); // Re-enable buttons
    }
  };

  // --- Render Logic ---
  if (loadingProject) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 5 }}><CircularProgress /></Box>;
  }
  if (!project) {
    return <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4, p: 2 }}><Alert severity="error">{error || 'Could not load project details.'}</Alert></Box>;
  }

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4, p: 2 }}>
      {/* Back Button */}
      <Button variant="outlined" onClick={() => navigate(-1)} sx={{ mb: 2 }}>
        Back
      </Button>
      <Typography variant="h4" gutterBottom>
        Training Settings for {project.name}
      </Typography>

      {/* Display general project fetch error if any */}
      {error && !loadingTemplate && !settings && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Card sx={{ mb: 4 }}>
        <CardContent>
          {/* --- Base Model Selection, Epochs, Repeats --- */}
          <Grid container spacing={2} alignItems="center" sx={{ mb: 2 }}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth required variant="outlined">
                <InputLabel id="base-model-label">Base Model</InputLabel>
                <Select
                  labelId="base-model-label"
                  id="base-model-select"
                  value={baseModel}
                  label="Base Model"
                  onChange={(e) => setBaseModel(e.target.value as string)}
                  disabled={loadingProject || training}
                >
                  <MenuItem value="" disabled><em>Select a base model</em></MenuItem>
                  {availableBaseModels.map((modelName) => (
                    <MenuItem key={modelName} value={modelName}>{modelName}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* --- Moved Epochs and Repeats --- */}
            {settings && (
              <Grid item xs={6} sm={3}>
                {renderSettingInput('max_train_epochs', settings.max_train_epochs)}
              </Grid>
            )}
            <Grid item xs={6} sm={3}>
              <TextField
                label="Repeats"
                type="number"
                value={repeats}
                onChange={(e) => setRepeats(parseInt(e.target.value, 10) || 1)}
                fullWidth
                variant="outlined"
                size="small"
                InputProps={{ inputProps: { min: 1 } }}
                disabled={loadingProject || training}
              />
            </Grid>
            {/* --- End Moved --- */}
          </Grid>

          {loadingTemplate && <CircularProgress size={24} sx={{ mt: 1 }} />}
          {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}

          {/* --- Advanced Settings Accordion --- */}
          {settings && (
            <Accordion
              defaultExpanded={false} // Set defaultExpanded to false
              sx={{ mt: 2, '.MuiAccordionSummary-content': { alignItems: 'center' } }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls="advanced-settings-content"
                id="advanced-settings-header"
              >
                <Typography variant="h6" sx={{ mr: 1 }}>Advanced Training Parameters</Typography>
                {/* Optional: Show a few key settings here if needed */}
                <Chip label={`LR: ${settings.learning_rate}`} size="small" variant="outlined" sx={{ mr: 0.5 }} />
                <Chip label={`Dim: ${settings.network_dim}/${settings.network_alpha}`} size="small" variant="outlined" sx={{ mr: 0.5 }} />
                <Chip label={`Batch: ${settings.train_batch_size}`} size="small" variant="outlined" />
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {/* Render all other settings here */}
                  {Object.entries(settings)
                    .filter(([key]) => !
                      ['pretrained_model_name_or_path', 'train_data_dir', 'output_dir', 'logging_dir',
                        'output_name', 'log_prefix', 'baseModelName', // Removed 'sample_prompts' from ignore list
                        'max_train_epochs' // Already moved
                      ].includes(key)
                    )
                    .map(([key, value]) => (
                      <Grid item xs={12} sm={6} md={4} key={key}>
                        {renderSettingInput(key, value)}
                      </Grid>
                    ))}
                </Grid>
              </AccordionDetails>
            </Accordion>
          )}

          {/* Start Training Button */}
          <Box sx={{ mt: 3, textAlign: 'right' }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleStartTraining}
              disabled={!baseModel || !settings || training || loadingTemplate || isPolling}
            >
              {training ? <CircularProgress size={24} /> : 'Start Training'}
            </Button>
          </Box>

          {/* Cancel Training Button - Appears only when training is running */}
          {isPolling && (trainingStatus === 'training' || trainingStatus === 'running') && (
            <Box sx={{ mt: 1, textAlign: 'right' }}>
              <Button
                variant="outlined"
                color="error"
                onClick={handleCancelTraining} // Define this function
                disabled={training} // Disable if a cancel request is already in progress
              >
                Cancel Training
              </Button>
            </Box>
          )}

        </CardContent>
      </Card>

      {/* --- Status and Log Display --- */}
      {(trainingStatus && trainingStatus !== 'starting') && (
        <Paper elevation={2} sx={{ mt: 3, p: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="h6">Training Status</Typography>
            <Chip
              label={trainingStatus.toUpperCase()}
              color={
                trainingStatus === 'completed' ? 'success' :
                  trainingStatus === 'error' ? 'error' :
                    trainingStatus === 'training' || trainingStatus === 'running' ? 'info' :
                      'default'
              }
              size="small"
            />
          </Box>
          <Box
            ref={logContainerRef}
            component="pre"
            sx={{
              bgcolor: 'grey.900',
              color: 'grey.200',
              p: 1.5,
              borderRadius: 1,
              maxHeight: '300px',
              overflowY: 'auto',
              whiteSpace: 'pre-wrap', // Wrap long lines
              wordBreak: 'break-all', // Break long words
              fontFamily: 'monospace',
              fontSize: '0.8rem'
            }}
          >
            {logContent || (isPolling ? 'Waiting for logs...' : 'No logs to display.')}
          </Box>
        </Paper>
      )}
      {/* --- End Status and Log Display --- */}

    </Box>
  );
};

export default TrainingSettings; 