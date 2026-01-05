import base64
import requests
import os

def fetch_mermaid_image():
    # Mermaid definition (from architecture_diagram.md)
    mermaid_code = """
graph TD
    subgraph Input Data
        fMRI[fMRI Voxel Data]
        NSD[Natural Scenes Dataset] --> fMRI
        NSD --> Images[Stimulus Images]
        NSD --> Captions[Image Captions]
    end

    subgraph Feature Extraction
        Images -->|Visual Encoder| CLIP_Img[CLIP Image Embeds]
        Captions -->|Text Encoder| CLIP_Txt[CLIP Text Embeds]
        Images -->|VAE Encoder| Latents[Image Latents]
    end

    subgraph "NH-LDM Training (Dual Stream)"
        direction LR
        
        subgraph "Semantic Stream (Stream A)"
            HVC[HVC Voxels] -->|Select Voxels| HVC_Sel
            HVC_Sel -->|Ridge Regression| SemanticDecoder[Semantic Decoder]
            SemanticDecoder -.->|Predicts| CLIP_Pred[Predicted CLIP Embeds]
            CLIP_Txt -.->|Target| SemanticDecoder
        end

        subgraph "Structural Stream (Stream B)"
            EVC[EVC Voxels] -->|Select Voxels| EVC_Sel
            EVC_Sel -->|Ridge Regression| StructuralDecoder[Structural Decoder]
            StructuralDecoder -.->|Predicts| Latents_Pred[Predicted VAE Latents]
            Latents -.->|Target| StructuralDecoder
        end
        
        fMRI --> HVC
        fMRI --> EVC
    end

    subgraph "Inference / Reconstruction"
        CLIP_Pred -->|Conditioning| LDM[Latent Diffusion Model]
        Latents_Pred -->|Initial Noise| LDM
        LDM -->|Denoising| ReconLatents[Reconstructed Latents]
        ReconLatents -->|VAE Decoder| FinalImage[Reconstructed Image]
    end
    """
    
    # Encode
    graphbytes = mermaid_code.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    
    url = "https://mermaid.ink/img/" + base64_string
    
    print(f"Fetching from {url}...")
    response = requests.get(url)
    
    output_path = r"C:\Users\PC\.gemini\antigravity\brain\6521599c-c96e-4393-98c2-31c1f424101c\architecture_diagram.png"
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved architecture diagram to {output_path}")
    else:
        print(f"Failed to fetch image. Status: {response.status_code}")

if __name__ == "__main__":
    fetch_mermaid_image()
