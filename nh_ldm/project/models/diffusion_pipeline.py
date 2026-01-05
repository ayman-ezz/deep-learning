import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class ReconstructionPipeline:
    def __init__(self, config):
        self.device = config['diffusion']['device']
        model_id = config['diffusion']['model_id']
        
        logger.info(f"Loading Diffusion Pipeline: {model_id}")
        # Load SD 1.4
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        # Use DDIM Scheduler as requested
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Disable safety checker for research/medical data (common practice for fMRI datasets)
        # to avoid false positives on noise/textures
        if hasattr(self.pipe, 'safety_checker'):
            self.pipe.safety_checker = None

        self.num_inference_steps = config['diffusion']['num_inference_steps']
        self.guidance_scale = config['diffusion']['guidance_scale']
        self.strength = config['diffusion']['strength']

    @torch.no_grad()
    def reconstruct(self, z_hat, c_hat):
        """
        Reconstruct image from structural (z_hat) and semantic (c_hat) latents.
        z_hat: [B, 4, 64, 64]
        c_hat: [B, 77, 768]
        """
        batch_size = z_hat.shape[0]
        
        # 1. Prepare Timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps)
        
        # Calculate start timestep based on strength
        # strength=1 -> start at T (pure noise)
        # strength=0 -> start at 0 (no noise)
        init_timestep = int(self.num_inference_steps * self.strength)
        timesteps = self.pipe.scheduler.timesteps[-init_timestep:]
        t_start = timesteps[0]
        
        # 2. Add noise to z_hat (Structural conditioning)
        # Scale z_hat match SD latent distribution expected std
        # SD 1.4 v1-4 vae scaling factor is typically 0.18215
        # Assuming z_hat is already in that space or trained to be in that space.
        # If the structural decoder learns to map to VAE latents *as they are*, 
        # then we just use them (maybe ensuring they are multiplied by 0.18215 if the target was unscaled).
        # Standard practice: VAE encodes to (z * 0.18215). So target Y was likely scaled.
        # We assume z_hat is predicted "ready-to-use" latent.
        
        noise = torch.randn_like(z_hat, device=self.device, dtype=self.pipe.unet.dtype)
        latents = self.pipe.scheduler.add_noise(z_hat, noise, t_start)
        
        # 3. Prepare Text Embeddings (Semantic conditioning)
        # c_hat is our "positive" prompt embedding
        # We need unconditional embedding for Classifier-Free Guidance
        uncond_input = self.pipe.tokenizer(
            [""] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
        )
        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concatenate for classifier-free guidance
        prompt_embeds = torch.cat([uncond_embeddings, c_hat])
        
        # 4. Denoising Loop
        for t in timesteps:
            # expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds
            ).sample
            
            # Perform CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 5. Decode latents to Image
        latents = 1 / 0.18215 * latents
        image = self.pipe.vae.decode(latents).sample
        
        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (image * 255).round().astype("uint8")
        
        return [Image.fromarray(img) for img in images]
