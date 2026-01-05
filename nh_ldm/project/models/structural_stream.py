import torch
import os
import logging
import pickle
from .ridge_decoder import RidgeRegression

logger = logging.getLogger(__name__)

class StructuralDecoder:
    """
    Stream B: Decodes Early Visual Cortex (EVC) to VAE Latents.
    Input: EVC Voxels
    Output: Latents (4, 64, 64) flattened
    """
    def __init__(self, config):
        self.config = config
        self.alphas = config['models']['ridge']['alphas']
        self.model = RidgeRegression(alphas=self.alphas)
        self.roi_names = config['data']['rois']['evc']
        # Usually we use all EVC voxels, or select
        self.voxel_indices = None

    def train(self, dataloader, vae_latents, val_dataloader=None, val_latents=None):
        logger.info("Starting Structural Decoder Training...")
        
        X_train = dataloader.dataset.fmri_data.numpy()
        Y_train = vae_latents.numpy()
        
        X_val = None
        Y_val = None
        if val_dataloader:
            X_val = val_dataloader.dataset.fmri_data.numpy()
            Y_val = val_latents.numpy()
            
        # Flatten Y (B, 4, 64, 64) -> (B, 16384)
        Y_train = Y_train.reshape(Y_train.shape[0], -1)
        if Y_val is not None:
            Y_val = Y_val.reshape(Y_val.shape[0], -1)
            
        # Fit
        self.model.fit(X_train, Y_train, X_val, Y_val)
        logger.info("Structural Decoder Training Complete.")

    def predict(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        pred_flat = self.model.predict(X_np)
        
        # Reshape to VAE latent shape [B, 4, 64, 64]
        return pred_flat.reshape(pred_flat.shape[0], 4, 64, 64)

    def save(self, output_dir):
        path = os.path.join(output_dir, 'structural_decoder.pkl')
        self.model.save(path)
