import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardActions,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Divider,
  CircularProgress,
  IconButton,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { LoraType, LoraProject } from '../types';

interface ProjectCardProps {
  type: LoraType;
  title: string;
  description: string;
  selected: boolean;
  onClick: () => void;
}

const ProjectCard: React.FC<ProjectCardProps> = ({ title, description, selected, onClick }) => (
  <Card
    onClick={onClick}
    sx={{
      cursor: 'pointer',
      border: selected ? '2px solid #2196f3' : '2px solid transparent',
      '&:hover': { borderColor: selected ? '#2196f3' : '#666' }
    }}
  >
    <CardContent>
      <Typography variant="h6">{title}</Typography>
      <Typography variant="body2" color="text.secondary">{description}</Typography>
    </CardContent>
  </Card>
);

const CreateLora: React.FC = () => {
  const navigate = useNavigate();
  const [selectedType, setSelectedType] = useState<LoraType | null>(null);
  const [name, setName] = useState('');
  const [existingProjects, setExistingProjects] = useState<LoraProject[]>([]);
  const [loading, setLoading] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<string | null>(null);

  useEffect(() => {
    fetchExistingProjects();
  }, []);

  const fetchExistingProjects = async () => {
    try {
      const response = await fetch('/api/projects/');
      if (response.ok) {
        const projects = await response.json();
        setExistingProjects(projects);
      } else {
        console.error('Failed to fetch projects:', response.statusText);
        setExistingProjects([]);
      }
    } catch (error) {
      console.error('Error fetching projects:', error);
      setExistingProjects([]);
    }
  };

  const handleCreate = async () => {
    if (!selectedType || !name.trim()) return;
    setLoading(true);
    try {
      const response = await fetch('/api/projects/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, type: selectedType })
      });

      if (response.ok) {
        const project = await response.json();
        navigate(`/upload/${project.id}`);
      } else {
        console.error('Error creating project:', await response.text());
      }
    } catch (error) {
      console.error('Error creating project:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDeleteDialog = (projectId: string) => {
    setProjectToDelete(projectId);
    setOpenDeleteDialog(true);
  };

  const handleCloseDeleteDialog = () => {
    setOpenDeleteDialog(false);
    setProjectToDelete(null);
  };

  const handleDeleteProject = async () => {
    if (!projectToDelete) return;
    try {
      const response = await fetch(`/api/projects/${projectToDelete}`, {
        method: 'DELETE',
      });
      if (response.ok || response.status === 204) {
        fetchExistingProjects();
      } else {
        console.error('Error deleting project:', response.statusText);
      }
    } catch (error) {
      console.error('Error deleting project:', error);
    } finally {
      handleCloseDeleteDialog();
    }
  };

  const handleLoadProject = (project: LoraProject) => {
    switch (project.training_status) {
      case 'created':
      case 'uploading':
        navigate(`/upload/${project.id}`);
        break;
      case 'tagging':
      case 'training':
      case 'completed':
      default:
        navigate(`/train/${project.id}`);
        break;
    }
  };

  return (
    <Box p={4} maxWidth={1200} mx="auto">
      <Typography variant="h4" mb={4}>Create your LoRA</Typography>

      <Typography variant="h6" mb={2}>Choose your LoRA type</Typography>
      <Grid container spacing={2} mb={4}>
        <Grid item xs={12} sm={4}>
          <ProjectCard
            type={LoraType.CHARACTER}
            title="Character"
            description="A specific person or character"
            selected={selectedType === LoraType.CHARACTER}
            onClick={() => setSelectedType(LoraType.CHARACTER)}
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <ProjectCard
            type={LoraType.STYLE}
            title="Style"
            description="A time period, art style, or general look"
            selected={selectedType === LoraType.STYLE}
            onClick={() => setSelectedType(LoraType.STYLE)}
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <ProjectCard
            type={LoraType.CONCEPT}
            title="Concept"
            description="Objects, clothing, anatomy, poses, etc."
            selected={selectedType === LoraType.CONCEPT}
            onClick={() => setSelectedType(LoraType.CONCEPT)}
          />
        </Grid>
      </Grid>

      <Box mb={4}>
        <Typography variant="subtitle1" mb={1}>Name your LoRA</Typography>
        <TextField
          fullWidth
          value={name}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
          placeholder="Enter a name for your LoRA"
        />
      </Box>

      <Button
        variant="contained"
        fullWidth
        onClick={handleCreate}
        disabled={!selectedType || !name.trim() || loading}
        sx={{ mb: 4 }}
      >
        {loading ? <CircularProgress size={24} /> : 'NEXT'}
      </Button>

      <Divider sx={{ my: 4 }} />

      <Typography variant="h5" mb={3}>Existing Projects</Typography>
      <Grid container spacing={2}>
        {existingProjects.map((project: LoraProject) => (
          <Grid item xs={12} sm={6} md={4} key={project.id}>
            <Card
              sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}
            >
              <CardContent
                sx={{ flexGrow: 1, cursor: 'pointer', '&:hover': { backgroundColor: 'action.hover' } }}
                onClick={() => handleLoadProject(project)}
              >
                <Typography variant="h6">{project.name}</Typography>
                <Typography variant="body2" color="text.secondary">
                  Type: {project.type}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Status: {project.training_status || 'New'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Images: {project.images?.length ?? 0}
                </Typography>
                {project.training_progress && (
                  <Typography variant="body2" color="text.secondary">
                    Progress: {project.training_progress.current_epoch}/{project.training_progress.total_epochs} epochs
                  </Typography>
                )}
                <Typography variant="body2" color="text.secondary">
                  Last modified: {new Date(project.last_modified || project.created_at).toLocaleDateString()}
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: 'flex-end' }}>
                <IconButton
                  aria-label="delete project"
                  onClick={(e) => { e.stopPropagation(); handleOpenDeleteDialog(project.id); }}
                  size="small"
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog
        open={openDeleteDialog}
        onClose={handleCloseDeleteDialog}
      >
        <DialogTitle>Delete Project?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this project and all its associated data (images, metadata, logs)? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDeleteDialog}>Cancel</Button>
          <Button onClick={handleDeleteProject} color="error" autoFocus>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CreateLora; 