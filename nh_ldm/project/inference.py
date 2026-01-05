import yaml
import torch
import torch.nn as nn
import os
import argparse
import numpy as np
from PIL import Image
from project.data.preprocessing import DataProcessor
from project.models.semantic_stream import SemanticDecoder
from project.models.structural_stream import StructuralDecoder
from project.models.diffusion_pipeline import ReconstructionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = config['diffusion']['device']

    # 1. Load Data
    logger.info("Loading Test Data...")
    processor = DataProcessor(config)
    test_dataset = processor.load_data(split='test')
    
    # We process average of repetitions if implemented, here simple load
    
    # 2. Load Models
    logger.info("Loading Models...")
    # Re-instantiate models and load weights
    semantic_model = SemanticDecoder(config)
    # Using load mechanism via pickle as saved in train.py
    with open("project/models/semantic_decoder.pkl", 'rb') as f:
        import pickle
        semantic_model.model = pickle.load(f)
        # Load indices 
    with open("project/models/semantic_indices.pkl", 'rb') as f:
        semantic_model.voxel_indices = pickle.load(f)

    structural_model = StructuralDecoder(config)
    with open("project/models/structural_decoder.pkl", 'rb') as f:
        structural_model.model = pickle.load(f)
        
    # 3. Load Pipeline
    logger.info("Loading Diffusion Pipeline...")
    pipeline = ReconstructionPipeline(config)
    
    # 4. Inference Loop
    logger.info("Starting Inference...")
    
    # Select ROIs
    hvc_data = processor.get_roi_data(test_dataset, 'hvc').fmri_data
    evc_data = processor.get_roi_data(test_dataset, 'evc').fmri_data
    
    # Batch processing
    batch_size = config['data']['batch_size']
    num_samples = len(test_dataset)
    
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch_hvc = hvc_data[i:end]
        batch_evc = evc_data[i:end]
        
        # Decode
        # c_hat: (B, 77, 768)
        c_hat = semantic_model.predict(batch_hvc) 
        c_hat = torch.from_numpy(c_hat).to(device).float()
        
        # z_hat: (B, 4, 64, 64)
        z_hat = structural_model.predict(batch_evc)
        z_hat = torch.from_numpy(z_hat).to(device).float()
        
        # Reconstruct
        # returns list of PIL Images
        images = pipeline.reconstruct(z_hat, c_hat)
        
        # Save
        for j, img in enumerate(images):
            idx = i + j
            save_path = os.path.join(args.output_dir, f"recon_{idx:04d}.png")
            img.save(save_path)
            logger.info(f"Saved image {idx+1}/{num_samples} to {save_path}")
            
    logger.info(f"Reconstruction complete. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
