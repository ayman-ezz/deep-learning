import numpy as np
import h5py
import os
import logging

logger = logging.getLogger(__name__)

class ROIExtractor:
    """
    Handles extraction of specific Regions of Interest (ROIs) from NSD data.
    """
    
    # Standard NSD-General ROI mappings (approximate/example values, 
    # dependent on specific mask file provided in NSD release)
    ROI_MAPPING = {
        'V1': 1,
        'V2': 2,
        'V3': 3,
        'V4': 4,
        'LOC': 5, # Lateral Occipital Complex
        'FFA': 6, # Fusiform Face Area
        'PPA': 7  # Parahippocampal Place Area
    }

    def __init__(self, roi_mask_path):
        """
        Initialize with path to the ROI mask file (e.g., .nii or .h5).
        For this implementation, we assume a flattened 1D array or similar h5 structure
        aligning with the beta weights.
        """
        self.roi_mask_path = roi_mask_path
        self.roi_data = self._load_mask()

    def _load_mask(self):
        """Loads the mask data."""
        if not os.path.exists(self.roi_mask_path):
            logger.warning(f"ROI mask file not found at {self.roi_mask_path}. Creating dummy mask.")
            # Dummy mask for testing/initialization if file missing
            np.random.seed(42) # Ensure consistency across runs
            return np.random.randint(0, 8, size=(15724,)) 

        try:
             # Example loading for H5; adapt for NIfTI if needed
            with h5py.File(self.roi_mask_path, 'r') as f:
                # Assuming dataset name 'roi' or similar
                keys = list(f.keys())
                return f[keys[0]][:]
        except Exception as e:
            logger.error(f"Error loading ROI mask: {e}")
            raise

    def get_indices(self, roi_names):
        """
        Returns indices of voxels belonging to the specified list of ROI names.
        """
        if isinstance(roi_names, str):
            roi_names = [roi_names]
            
        indices = []
        for name in roi_names:
            if name in self.ROI_MAPPING:
                roi_val = self.ROI_MAPPING[name]
                # Find indices where mask equals the ROI value
                roi_indices = np.where(self.roi_data == roi_val)[0]
                indices.append(roi_indices)
            else:
                logger.warning(f"ROI {name} not found in mapping.")
        
        if not indices:
            return np.array([], dtype=int)
            
        return np.concatenate(indices)

    def select_voxels(self, data, roi_indices):
        """
        Selects columns from data corresponding to the indices.
        Data shape expected: (num_samples, num_voxels)
        """
        if data.shape[1] < np.max(roi_indices):
             logger.warning("Max ROI index exceeds data dimensions.")
             
        return data[:, roi_indices]
