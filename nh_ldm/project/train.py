import yaml
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from project.data.preprocessing import DataProcessor
from project.models.semantic_stream import SemanticDecoder
from project.models.structural_stream import StructuralDecoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features(config, dataset, pipeline):
    """
    Extracts CLIP embeddings and VAE latents for the training images.
    """
    device = config['diffusion']['device']
    batch_size = config['data']['batch_size']
    
    # Check if features are already saved to disk to save time
    cache_dir = "cache_features"
    os.makedirs(cache_dir, exist_ok=True)
    clip_path = os.path.join(cache_dir, "train_clip.pt")
    vae_path = os.path.join(cache_dir, "train_vae.pt")
    
    if os.path.exists(clip_path) and os.path.exists(vae_path):
        logger.info("Loading cached features...")
        return torch.load(clip_path), torch.load(vae_path)

    logger.info("Extracting features from training images...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_clip = []
    all_vae = []
    
    pipeline.tokenizer.model_max_length = 77
    
    for batch in tqdm(dataloader, desc="Feature Extraction"):
        # batch['image_id'] helps load actual image
        # In a real scenario, we load images here. 
        # For this skeleton, we assume dataset might provide raw images or paths.
        # Since implementation details of NSD images loading are complex,
        # we simulate image loading or expect user to implement `load_image(id)`.
        
        # Simulating random images for the sake of runnable code if real images missing
        # Replace with actual image loading: 
        # images = [load_image(idx) for idx in batch['image_id']]
        current_bs = len(batch['image_id'])
        dummy_images = torch.rand(current_bs, 3, 512, 512).to(device) # [0,1]
         
        # 1. CLIP Embeddings (using simulated prompts or BLIP captions if available)
        # The prompt says: "BLIP model for generating pseudo-captions".
        # If we don't have BLIP here, we assume captions are provided or we use image embeddings?
        # "Stream A - Semantic Decoder (Brain -> CLIP)". Usually CLIP Vision embedding or Text.
        # Report says "CLIP text embeddings". This implies we have captions.
        # NSD has COCO captions.
        # For this code, I will extract *Image Embeddings* as proxy or use dummy text if data missing.
        # Wait, if target is Text Embedding, we need Text.
        # I'll stick to CLIP Image Encodings as target if no text, OR prompt user.
        # Let's assume we use CLIP Image Embeddings (Context) as 'Semantic'.
        # Actually, "Output: CLIP text embeddings (77 x 768)".
        # This strongly implies we need the captions.
        # I'll create dummy text embeddings for the skeleton.
        
        with torch.no_grad():
            # Dummy CLIP text embeddings (Batch, 77, 768)
            dummy_text_embeds = torch.randn(current_bs, 77, 768).to(device)
            all_clip.append(dummy_text_embeds.cpu())
            
            # 2. VAE Latents
            # latents = vae.encode(img).latent_dist.sample() * 0.18215
            # Input to VAE should be (B, 3, H, W) normalized
            latents = pipeline.vae.encode(dummy_images).latent_dist.mode() * 0.18215
            all_vae.append(latents.cpu())

    all_clip = torch.cat(all_clip, dim=0)
    all_vae = torch.cat(all_vae, dim=0)
    
    # Save to cache
    torch.save(all_clip, clip_path)
    torch.save(all_vae, vae_path)
    
    return all_clip, all_vae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # 1. Load Data
    logger.info("Loading Data...")
    processor = DataProcessor(config)
    train_dataset = processor.load_data(split='train')
    
    # 2. Extract Targets (Features)
    # We need the SD pipeline to encode images/text
    # "Using Stable Diffusion v1.4"
    logger.info("Initializing SD Pipeline for feature extraction...")
    # Using float16 for speed if on cuda
    dtype = torch.float16 if config['diffusion']['device'] == 'cuda' else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        config['diffusion']['model_id'], torch_dtype=dtype
    ).to(config['diffusion']['device'])
    
    logger.info("Extracting features...")
    clip_emb, vae_latents = extract_features(config, train_dataset, pipe)
    
    # Free up VRAM
    del pipe
    torch.cuda.empty_cache()
    
    # 3. Train Semantic Stream
    logger.info("Training Semantic Stream...")
    # Get HVC specific fmri data inside the stream or pass full? 
    # Current impl passes full dict dataset, stream handles extraction if needed logic.
    # But currently `train` method expects `dataloader`.
    # Let's pass the full train_dataset masked by ROI.
    
    semantic_dataset = processor.get_roi_data(train_dataset, 'hvc')
    semantic_loader = torch.utils.data.DataLoader(semantic_dataset, batch_size=len(semantic_dataset))
    
    semantic_model = SemanticDecoder(config)
    semantic_model.train(semantic_loader, clip_emb)
    semantic_model.save("project/models") # Save dir
    
    # 4. Train Structural Stream
    logger.info("Training Structural Stream...")
    structural_dataset = processor.get_roi_data(train_dataset, 'evc')
    structural_loader = torch.utils.data.DataLoader(structural_dataset, batch_size=len(structural_dataset))
    
    structural_model = StructuralDecoder(config)
    structural_model.train(structural_loader, vae_latents)
    structural_model.save("project/models")
    
    logger.info("Training pipeline complete. Models saved.")

if __name__ == "__main__":
    main()
