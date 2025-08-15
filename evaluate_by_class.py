#!/usr/bin/env python3
"""
Class-Aware Model Evaluation Script
===================================

This script evaluates a trained segmentation model on a validation set,
providing separate performance metrics for each defect class (e.g., 'chip', 'check').

Usage:
    python evaluate_by_class.py --model_path "path/to/best_model.pth" \
                                --dataset_dir "path/to/segmentation_dataset" \
                                --yolo_labels_dir "path/to/original/yolo/labels/val" \
                                --output_dir "path/to/class_evaluation_results"
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from collections import defaultdict

# Import necessary components from your existing scripts
from train_segmentation_model import DefectSegmentationDataset, collate_fn_skip_corrupted

def calculate_metrics(pred_tensor, target_tensor, threshold=0.5):
    """Calculates segmentation metrics for a batch of predictions."""
    pred_binary = (torch.sigmoid(pred_tensor) > threshold).float()
    
    # Flatten tensors for metric calculation
    pred_flat = pred_binary.view(-1)
    target_flat = target_tensor.view(-1)
    
    # True Positives, False Positives, False Negatives
    tp = (pred_flat * target_flat).sum().item()
    fp = ((1 - target_flat) * pred_flat).sum().item()
    fn = (target_flat * (1 - pred_flat)).sum().item()
    
    # Intersection over Union (IoU)
    iou = tp / (tp + fp + fn + 1e-6)
    
    # Dice Coefficient
    dice = (2.0 * tp) / (2 * tp + fp + fn + 1e-6)
    
    return {'iou': iou, 'dice': dice}

def save_visual_comparison(image_tensor, target_mask, pred_mask, output_path, threshold=0.5):
    """Saves a side-by-side visual comparison of the results."""
    
    # Denormalize image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * std + mean) * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV

    # Convert masks to 3-channel BGR images
    target_mask = (target_mask.cpu().numpy().squeeze() * 255).astype(np.uint8)
    target_mask_bgr = cv2.cvtColor(target_mask, cv2.COLOR_GRAY2BGR)
    
    pred_mask_binary = (torch.sigmoid(pred_mask) > threshold).float()
    pred_mask = (pred_mask_binary.cpu().numpy().squeeze() * 255).astype(np.uint8)
    pred_mask_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

    # Create an overlay of the prediction on the original image
    overlay = image.copy()
    overlay[pred_mask > 0] = [0, 0, 255]  # Highlight prediction in red
    overlayed_image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    # Concatenate images for comparison
    comparison_img = np.concatenate((image, target_mask_bgr, pred_mask_bgr, overlayed_image), axis=1)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison_img, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison_img, 'Ground Truth', (image.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison_img, 'Prediction', (image.shape[1]*2 + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison_img, 'Overlay', (image.shape[1]*3 + 10, 30), font, 1, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), comparison_img)

def get_defect_classes_from_yolo(yolo_label_path: Path) -> set:
    """Reads a YOLO annotation file and returns a set of class IDs present."""
    if not yolo_label_path.exists():
        return {'no_defect'} # Corresponds to clean images
    
    classes = set()
    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()
    
    if not lines or all(line.strip() == '' for line in lines):
        return {'no_defect'}

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            if class_id == 0:
                classes.add('chip_defect')
            elif class_id == 1:
                classes.add('check_defect')
    
    return classes if classes else {'no_defect'}


def main():
    parser = argparse.ArgumentParser(description='Evaluate a segmentation model with class-specific metrics.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth file).')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the prepared segmentation dataset directory.')
    parser.add_argument('--yolo_labels_dir', type=str, required=True, help='Path to the original YOLO .txt label directory for the validation set.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results.')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to evaluate on (must match yolo_labels_dir).')
    
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    yolo_labels_dir = Path(args.yolo_labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {device}")

    # --- Load Model ---
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_architecture = config.get('model_architecture', 'Unet')
    encoder_name = config.get('encoder_name', 'resnet34')
    
    print(f"Re-creating model: {model_architecture} with encoder {encoder_name}")
    model_class = getattr(smp, model_architecture) if hasattr(smp, model_architecture) else smp.Unet
    model = model_class(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # --- Load Data ---
    eval_dataset = DefectSegmentationDataset(
        images_dir=Path(args.dataset_dir) / args.split / 'images',
        masks_dir=Path(args.dataset_dir) / args.split / 'masks',
        augment=False
    )
    
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0) # Use batch_size=1 for per-image analysis
    
    print(f"Found {len(eval_dataset)} samples in the '{args.split}' split.")

    # --- Evaluation Loop ---
    class_metrics = defaultdict(lambda: {'iou': 0.0, 'dice': 0.0, 'count': 0})

    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(eval_loader, desc=f"Evaluating by class")):
            if image is None: continue

            # Get the filename to find the corresponding YOLO label
            filename_stem = eval_dataset.valid_files[i]
            yolo_label_path = yolo_labels_dir / f"{filename_stem}.txt"
            
            # Determine the classes present in this image
            defect_classes = get_defect_classes_from_yolo(yolo_label_path)

            # Run inference
            image = image.to(device)
            mask = mask.to(device)
            pred = model(image)
            
            # Calculate metrics for this single image
            metrics = calculate_metrics(pred, mask)
            
            # Accumulate metrics for each class present in the image
            for defect_class in defect_classes:
                class_metrics[defect_class]['iou'] += metrics['iou']
                class_metrics[defect_class]['dice'] += metrics['dice']
                class_metrics[defect_class]['count'] += 1

                # Save visual results into class-specific folders
                class_visuals_dir = output_dir / defect_class
                class_visuals_dir.mkdir(exist_ok=True)
                output_path = class_visuals_dir / f"{filename_stem}_comparison.png"
                save_visual_comparison(image.squeeze(0), mask.squeeze(0), pred.squeeze(0), output_path)

    # --- Final Results ---
    final_results = {}
    print("\n--- Class-Specific Evaluation Complete ---")
    for defect_class, data in class_metrics.items():
        count = data['count']
        if count > 0:
            avg_iou = data['iou'] / count
            avg_dice = data['dice'] / count
            final_results[defect_class] = {'avg_iou': avg_iou, 'avg_dice': avg_dice, 'sample_count': count}
            print(f"\nClass: {defect_class}")
            print(f"  Sample Count: {count}")
            print(f"  Average IoU:  {avg_iou:.4f}")
            print(f"  Average Dice: {avg_dice:.4f}")
    print("------------------------------------------")
    
    # Save results to a JSON file
    with open(output_dir / 'class_evaluation_metrics.json', 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print(f"Detailed metrics saved to: {output_dir / 'class_evaluation_metrics.json'}")
    print(f"Visual comparisons saved in class-specific subfolders inside: {output_dir}")

if __name__ == '__main__':
    main()

