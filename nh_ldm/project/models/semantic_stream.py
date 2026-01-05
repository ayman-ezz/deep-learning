import torch
import os
import logging
import pickle
from .ridge_decoder import RidgeRegression, select_top_voxels

logger = logging.getLogger(__name__)

class SemanticDecoder:
    """
    Stream A: Decodes Header Visual Cortex (HVC) to CLIP text embeddings.
    """
    def __init__(self, config):
        self.config = config
        self.alphas = config['models']['ridge']['alphas']
        self.n_voxels = config['models']['ridge']['n_voxels_semantic']
        self.model = RidgeRegression(alphas=self.alphas)
        self.roi_names = config['data']['rois']['hvc']
        self.voxel_indices = None # Selected during training

    def train(self, dataloader, clip_embeddings, val_dataloader=None, val_embeddings=None):
        """
        X: HVC fMRI Data
        Y: CLIP Text Embeddings (e.g. 77x768 flattened)
        """
        logger.info("Starting Semantic Decoder Training...")
        
        # Extract features from dataloader
        # For simplicity, assuming dataloader yields full batch or we aggregate
        # In real scaling, we'd handle batches but Ridge needs full matrix often
        
        # 1. Aggregate Data
        X_train = dataloader.dataset.fmri_data.numpy() # Warning: Memory
        Y_train = clip_embeddings.numpy()
        
        X_val = None
        Y_val = None
        if val_dataloader:
            X_val = val_dataloader.dataset.fmri_data.numpy()
            Y_val = val_embeddings.numpy()

        # 2. Select Features (HVC + Reliablity)
        # Assuming DataProcessor already filtered for ROIs or we do it here?
        # Let's assume input X is *all* fMRI and we filtered by ROI in DataProcessor
        # If X is full brain, we need global indices. 
        # For now, assuming X passed here is already specific to HVC preference via data config
        
        if X_train.shape[1] > self.n_voxels:
            logger.info(f"Selecting top {self.n_voxels} voxels...")
            self.voxel_indices = select_top_voxels(X_train, Y_train, keep=self.n_voxels)
            X_train = X_train[:, self.voxel_indices]
            if X_val is not None:
                 X_val = X_val[:, self.voxel_indices]
        
        # Flatten Y if needed (CLIP 77x768 -> 59136)
        # Ridge handles multi-output
        original_y_shape = Y_train.shape[1:]
        Y_train = Y_train.reshape(Y_train.shape[0], -1)
        if Y_val is not None:
             Y_val = Y_val.reshape(Y_val.shape[0], -1)

        # 3. Fit
        self.model.fit(X_train, Y_train, X_val, Y_val)
        
        logger.info("Semantic Decoder Training Complete.")

    def predict(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        
        # Apply voxel selection
        if self.voxel_indices is not None:
            X_np = X_np[:, self.voxel_indices]
            
        pred_flat = self.model.predict(X_np)
        
        # Reshape to CLIP dimensions [B, 77, 768]
        return pred_flat.reshape(pred_flat.shape[0], 77, 768)

    def save(self, output_dir):
        path = os.path.join(output_dir, 'semantic_decoder.pkl')
        self.model.save(path)
        # Also need to save voxel_indices
        with open(os.path.join(output_dir, 'semantic_indices.pkl'), 'wb') as f:
            pickle.dump(self.voxel_indices, f)
