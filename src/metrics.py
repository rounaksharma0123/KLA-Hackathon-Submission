import os
import cv2
import torch
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import sys

def calculate_metrics(gt_dir, pred_dir, device='cpu'):
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    psnr_list, ssim_list, lpips_list = [], [], []
    
    files = sorted(os.listdir(pred_dir))
    valid_files_count = 0
    
    for f in files:
        if not f.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        pred_path = os.path.join(pred_dir, f)
        base_name = os.path.splitext(f)[0]
        
        # Look for the .npy file in the Ground Truth folder
        gt_path = os.path.join(gt_dir, base_name + '.npy')
        
        if not os.path.exists(gt_path):
            print(f"[ERROR] Missing GT for {f} at {gt_path}")
            continue
            
        # Load GT as NumPy array and convert to uint8 (0-255 format)
        gt_img = np.load(gt_path)
        if gt_img.dtype in [np.float32, np.float64]:
            if gt_img.max() <= 1.0:
                gt_img = (gt_img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                gt_img = gt_img.clip(0, 255).astype(np.uint8)
        else:
            gt_img = gt_img.astype(np.uint8)
            
        # Ensure GT is 2D (grayscale)
        if gt_img.ndim == 3:
            gt_img = gt_img.squeeze()
            
        # Load Prediction as standard image
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate PSNR and SSIM
        curr_psnr = psnr(gt_img, pred_img, data_range=255)
        curr_ssim = ssim(gt_img, pred_img, data_range=255)
        psnr_list.append(curr_psnr)
        ssim_list.append(curr_ssim)
        
        # LPIPS calculation
        gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
        pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)
        
        gt_tensor = torch.from_numpy(gt_img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        pred_tensor = torch.from_numpy(pred_img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        
        with torch.no_grad():
            curr_lpips = loss_fn_vgg(pred_tensor.to(device), gt_tensor.to(device)).item()
        lpips_list.append(curr_lpips)
        
        valid_files_count += 1
        
    if valid_files_count == 0:
        print("[ERROR] No valid comparisons could be made.")
        sys.exit(1)
        
    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)

if __name__ == "__main__":
    gt_dir = os.path.expanduser("~/Downloads/Data-public/train/GT")
    bicubic_dir = "../outputs/bicubic"
    restored_dir = "../outputs/restored"
    
    print("Calculating metrics for Bicubic Baseline...")
    b_psnr, b_ssim, b_lpips = calculate_metrics(gt_dir, bicubic_dir)
    print(f"Bicubic Baseline - PSNR: {b_psnr:.2f} dB, SSIM: {b_ssim:.4f}, LPIPS: {b_lpips:.4f}")
    
    print("\nCalculating metrics for Final Restored Model...")
    r_psnr, r_ssim, r_lpips = calculate_metrics(gt_dir, restored_dir)
    print(f"Final Restored - PSNR: {r_psnr:.2f} dB, SSIM: {r_ssim:.4f}, LPIPS: {r_lpips:.4f}")