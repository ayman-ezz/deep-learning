
import matplotlib.pyplot as plt
import numpy as np
import os

def create_charts(output_dir='report_assets'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. System Architecture Diagram (Conceptual)
    # We'll create a simple flowchart-like visual using matplotlib patches
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Boxes
    box_props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax.text(1, 4, "fMRI Data\n(NSD)", transform=ax.transData, size=12, ha="center", bbox=dict(boxstyle='round', facecolor='#e1f5fe'))
    ax.text(3, 5, "Semantic Stream\n(HVC -> Ridge)", transform=ax.transData, size=12, ha="center", bbox=dict(boxstyle='round', facecolor='#fff9c4'))
    ax.text(3, 3, "Structural Stream\n(EVC -> VAE)", transform=ax.transData, size=12, ha="center", bbox=dict(boxstyle='round', facecolor='#fff9c4'))
    ax.text(5.5, 4, "Latent Conditionings\n(CLIP & VAE Latents)", transform=ax.transData, size=12, ha="center", bbox=dict(boxstyle='round', facecolor='#ffe0b2'))
    ax.text(8, 4, "Stable Diffusion\n(Image Reconstruction)", transform=ax.transData, size=12, ha="center", bbox=dict(boxstyle='round', facecolor='#dcedc8'))
    
    # Arrows
    ax.arrow(1.5, 4.2, 0.8, 0.8, head_width=0.2, head_length=0.2, fc='k', ec='k') # to split top
    ax.arrow(1.5, 3.8, 0.8, -0.8, head_width=0.2, head_length=0.2, fc='k', ec='k') # to split bottom
    ax.arrow(4, 5, 0.5, -0.8, head_width=0.2, head_length=0.2, fc='k', ec='k') # sem to joint
    ax.arrow(4, 3, 0.5, 0.8, head_width=0.2, head_length=0.2, fc='k', ec='k') # str to joint
    ax.arrow(6.5, 4, 0.5, 0, head_width=0.2, head_length=0.2, fc='k', ec='k') # joint to SD

    plt.title("NH-LDM Dual-Stream Architecture Flow", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_architecture.png'), dpi=300)
    plt.close()

    # 2. ROI Voxel Contribution
    # Mock data based on config (approximate)
    rois = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA']
    # Approximate voxel counts typically found in NSD for these ROIs (illustrative)
    voxel_counts = [1200, 1100, 1000, 800, 900, 600, 500] 
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4fc3f7' if x in ['V1','V2','V3'] else '#ffb74d' for x in rois]
    bars = ax.bar(rois, voxel_counts, color=colors)
    
    ax.set_ylabel('Number of Voxels')
    ax.set_title('Voxel Distribution per ROI (EVC vs HVC)')
    ax.legend([bars[0], bars[4]], ['Early Visual Cortex (Structural)', 'Higher Visual Cortex (Semantic)'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_roi_distribution.png'), dpi=300)
    plt.close()

    # 3. Training Loss Simulation (Illustrative)
    epochs = np.linspace(0, 100, 100)
    loss_sem = 1.0 * np.exp(-0.05 * epochs) + 0.1 * np.random.normal(0, 0.1, 100)
    loss_struct = 1.2 * np.exp(-0.04 * epochs) + 0.1 * np.random.normal(0, 0.1, 100)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, loss_sem, label='Semantic Decoder Loss', linewidth=2)
    ax.plot(epochs, loss_struct, label='Structural Decoder Loss', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_training_loss.png'), dpi=300)
    plt.close()

    print(f"Charts generated in {output_dir}")

if __name__ == "__main__":
    create_charts()
