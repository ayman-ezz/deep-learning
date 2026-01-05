import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from .roi_masks import ROIExtractor
import logging

logger = logging.getLogger(__name__)

class NSDDataset(Dataset):
    def __init__(self, fmri_data, image_ids, transform=None):
        self.fmri_data = torch.FloatTensor(fmri_data)
        self.image_ids = image_ids # List or array of image indices/filenames
        self.transform = transform

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        # Returns fMRI data and the corresponding image ID (or image itself later)
        return {
            'fmri': self.fmri_data[idx],
            'image_id': self.image_ids[idx]
        }

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.nsd_path = config['data']['nsd_path']
        self.subject = config['data']['subjects'][0] # Single subject for now
        self.roi_extractor = ROIExtractor(
            os.path.join(self.nsd_path, self.subject, 'roi_mask.h5')
        )

    def load_data(self, split='train'):
        """
        Loads fMRI betas and stimuli for the given split (train/test).
        """
        # Paths (Placeholder structure based on standard NSD)
        fmri_path = os.path.join(self.nsd_path, self.subject, 'betas_all.h5')
        
        if not os.path.exists(fmri_path):
            logger.warning(f"Data file not found: {fmri_path}. Generating dummy data.")
            return self._generate_dummy_data(split)

        # Load fMRI data
        with h5py.File(fmri_path, 'r') as f:
            # Assuming shape (num_trials, num_voxels)
            all_betas = f['betas'][:] 
        
        # Simple split logic based on config counts
        # Real NSD has specific indices for shared 1000 test images
        n_train = self.config['data']['train_trials']
        n_test = self.config['data']['test_trials']
        
        if split == 'train':
            data = all_betas[:n_train]
            image_ids = np.arange(n_train) # Placeholder IDs
        elif split == 'test':
            # Assuming test set follows train set
            data = all_betas[n_train:n_train+n_test]
            image_ids = np.arange(n_train, n_train+n_test)
        else:
            raise ValueError("Split must be 'train' or 'test'")

        # Normalize
        data = self.normalize(data)

        return NSDDataset(data, image_ids)

    def normalize(self, data):
        """Z-score normalization."""
        if self.config['data']['normalization'] == 'zscore':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            # Avoid divide by zero
            std[std == 0] = 1.0
            return (data - mean) / std
        return data

    def get_roi_data(self, dataset, roi_name_key):
        """
        Extracts specific ROI voxels from the dataset.
        roi_name_key: 'evc' or 'hvc' (keys in config)
        """
        roi_names = self.config['data']['rois'][roi_name_key]
        indices = self.roi_extractor.get_indices(roi_names)
        
        # Access the underlying tensor in the dataset
        full_data = dataset.fmri_data.numpy()
        selected_data = self.roi_extractor.select_voxels(full_data, indices)
        
        # Return new dataset with selected voxels
        return NSDDataset(selected_data, dataset.image_ids)

    def _generate_dummy_data(self, split):
        """Generates random noise data for testing pipeline."""
        n_samples = self.config['data']['train_trials'] if split == 'train' else self.config['data']['test_trials']
        n_voxels = 15724 # Approx NSD general size
        
        logger.info(f"Generating dummy {split} data: ({n_samples}, {n_voxels})")
        data = np.random.randn(n_samples, n_voxels).astype(np.float32)
        image_ids = np.arange(n_samples)
        
        return NSDDataset(data, image_ids)
