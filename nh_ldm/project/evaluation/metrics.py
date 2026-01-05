import torch
import numpy as np
from PIL import Image
try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    # Fallback or stub if skimage not installed
    def ssim_func(im1, im2, channel_axis=None):
        return 0.0

def compute_correlation(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def n_way_accuracy(pred_vectors, target_vectors, n_way=2, num_trials=1000):
    """
    Computes N-way identification accuracy.
    pred_vectors: (N, D)
    target_vectors: (N, D)
    """
    pred_vectors = pred_vectors.cpu().numpy() if isinstance(pred_vectors, torch.Tensor) else pred_vectors
    target_vectors = target_vectors.cpu().numpy() if isinstance(target_vectors, torch.Tensor) else target_vectors
    
    num_samples = len(pred_vectors)
    correct = 0
    total = 0
    
    # Pre-compute correlation matrix (N x N)
    # Norm vectors
    pred_norm = pred_vectors / (np.linalg.norm(pred_vectors, axis=1, keepdims=True) + 1e-10)
    target_norm = target_vectors / (np.linalg.norm(target_vectors, axis=1, keepdims=True) + 1e-10)
    
    # similarity[i, j] = corr(pred[i], target[j])
    similarity_matrix = pred_norm @ target_norm.T
    
    for i in range(min(num_samples, num_trials)):
        true_score = similarity_matrix[i, i]
        
        # Select N-1 distractors
        distractors = np.random.choice([x for x in range(num_samples) if x != i], n_way - 1, replace=False)
        distractor_scores = similarity_matrix[i, distractors]
        
        # Check if true score is highest
        if true_score > np.max(distractor_scores):
            correct += 1
        total += 1
            
    return correct / total

def clip_similarity_score(image_embeds_recon, image_embeds_true):
    """
    Average cosine similarity between reconstructed and ground truth CLIP embeddings.
    """
    if isinstance(image_embeds_recon, torch.Tensor):
        image_embeds_recon = image_embeds_recon.cpu().numpy()
    if isinstance(image_embeds_true, torch.Tensor):
        image_embeds_true = image_embeds_true.cpu().numpy()
        
    scores = []
    for p, t in zip(image_embeds_recon, image_embeds_true):
        scores.append(compute_correlation(p, t))
        
    return np.mean(scores)

def ssim_score(image_recon, image_true):
    """
    Compute SSIM between two images (PIL or numpy).
    """
    if isinstance(image_recon, Image.Image):
        image_recon = np.array(image_recon)
    if isinstance(image_true, Image.Image):
        image_true = np.array(image_true)
        
    # Standardize size? Assumed same.
    # Gray scale or color? usually color channel_axis=2
    try:
        score = ssim_func(image_recon, image_true, channel_axis=2)
        return score
    except Exception as e:
        print(f"SSIM Error: {e}")
        return 0.0
