import matplotlib.pyplot as plt
import numpy as np

def visualize_results(ground_truth_images, reconstructed_images, output_path="comparison.png"):
    """
    Plots side-by-side comparison of Ground Truth vs Reconstruction.
    """
    n = len(ground_truth_images)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    
    if n == 1:
        axes = np.array([axes]).T
    
    for i in range(n):
        # Ground Truth
        axes[0, i].imshow(ground_truth_images[i])
        axes[0, i].set_title("Ground Truth")
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructed_images[i])
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
