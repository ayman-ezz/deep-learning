"""
Minimal test script to verify diffusion pipeline works.
"""
import yaml
import torch
import os
from project.models.diffusion_pipeline import ReconstructionPipeline
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override to reduce computation
    config['diffusion']['num_inference_steps'] = 10  # Reduce from 50 to 10 for testing
    
    logger.info("Loading pipeline...")
    pipeline = ReconstructionPipeline(config)
    
    device = config['diffusion']['device']
    
    # Create dummy latents
    logger.info("Creating dummy latents...")
    z_hat = torch.randn(1, 4, 64, 64).to(device).float()
    c_hat = torch.randn(1, 77, 768).to(device).float()
    
    # Run reconstruction
    logger.info("Running reconstruction...")
    try:
        images = pipeline.reconstruct(z_hat, c_hat)
        
        os.makedirs('results', exist_ok=True)
        images[0].save('results/test_output.png')
        logger.info("SUCCESS! Test image saved to results/test_output.png")
    except Exception as e:
        logger.error(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
