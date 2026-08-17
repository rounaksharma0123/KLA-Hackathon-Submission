import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def load_npy_as_img(path):
    img = np.load(path)
    if img.dtype in [np.float32, np.float64]:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    if img.ndim == 3:
        img = img.squeeze()
    return img

def create_visual_grid(base_name, noisy_dir, bicubic_dir, restored_dir, gt_dir, save_path):
    # Load .npy files for inputs/targets
    noisy = load_npy_as_img(os.path.join(noisy_dir, base_name + '.npy'))
    gt = load_npy_as_img(os.path.join(gt_dir, base_name + '.npy'))
    
    # Load .png files for model outputs
    bicubic = cv2.imread(os.path.join(bicubic_dir, base_name + '.png'), cv2.IMREAD_GRAYSCALE)
    restored = cv2.imread(os.path.join(restored_dir, base_name + '.png'), cv2.IMREAD_GRAYSCALE)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title("1. NoisyLR Input")
    axes[0].axis('off')
    
    axes[1].imshow(bicubic, cmap='gray')
    axes[1].set_title("2. Bicubic Baseline")
    axes[1].axis('off')
    
    axes[2].imshow(restored, cmap='gray')
    axes[2].set_title("3. Final Restored")
    axes[2].axis('off')
    
    axes[3].imshow(gt, cmap='gray')
    axes[3].set_title("4. Ground Truth")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved visual grid to {save_path}")

if __name__ == "__main__":
    noisy_dir = os.path.expanduser("~/Downloads/Data-public/train/NoisyLR")
    gt_dir = os.path.expanduser("~/Downloads/Data-public/train/GT")
    bicubic_dir = "../outputs/bicubic"
    restored_dir = "../outputs/restored"
    save_dir = "../results/samples"
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Just use the base number/name without extensions
    images_to_visualize = ["000010", "000019", "000031"] 
    
    for base_name in images_to_visualize:
        save_path = os.path.join(save_dir, f"{base_name}_grid.png")
        create_visual_grid(base_name, noisy_dir, bicubic_dir, restored_dir, gt_dir, save_path)