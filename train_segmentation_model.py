#!/usr/bin/env python3
"""
Advanced Segmentation Model Training Pipeline
============================================

This script trains a state-of-the-art U-Net model for chip and check defect detection
using the generated segmentation masks. It incorporates advanced techniques from
recent research papers for optimal performance.

Features:
- U-Net with pre-trained encoder (ResNet, EfficientNet, etc.)
- Compound loss function (Dice + Focal + BCE)
- Advanced data augmentation
- Learning rate scheduling
- Early stopping and model checkpointing
- Comprehensive evaluation metrics

Author: Manus AI Agent
Date: 2025-01-14
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefectSegmentationDataset(Dataset):
    """
    Dataset class for defect segmentation
    """
    
    def __init__(self, images_dir: str, masks_dir: str, transform=None, augment=False):
        """
        Initialize the dataset
        
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing segmentation masks
            transform: Transforms to apply to images and masks
            augment: Whether to apply data augmentation
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.augment = augment
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        # Filter to only include images that have corresponding masks
        self.valid_files = []
        for img_file in self.image_files:
            mask_file = self.masks_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                self.valid_files.append(img_file.stem)
        
        logger.info(f"Dataset initialized with {len(self.valid_files)} samples")
        
        # Setup augmentation pipeline
        if self.augment:
            self.augmentation = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.3
                ),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ])
        else:
            self.augmentation = None
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_name = self.valid_files[idx]
        
        # Try different extensions for image
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = self.images_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image file not found for {img_name}")
        
        mask_path = self.masks_dir / f"{img_name}.png"
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Normalize mask to 0-1
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Apply transforms
        if self.transform:
            # Convert to PIL format for torchvision transforms
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
            
            # Convert mask to tensor
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        else:
            # Default conversion to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask

class CombinedLoss(nn.Module):
    """
    Combined loss function using Dice Loss + Focal Loss + Binary Cross Entropy
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.3, bce_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for segmentation"""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def focal_loss(self, pred, target):
        """Focal loss for handling class imbalance"""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        total_loss = (self.dice_weight * dice + 
                     self.focal_weight * focal + 
                     self.bce_weight * bce)
        
        return total_loss, {
            'dice_loss': dice.item(),
            'focal_loss': focal.item(),
            'bce_loss': bce.item(),
            'total_loss': total_loss.item()
        }

class SegmentationTrainer:
    """
    Main trainer class for segmentation model
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = CombinedLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0),
            dice_weight=config.get('dice_weight', 0.5),
            focal_weight=config.get('focal_weight', 0.3),
            bce_weight=config.get('bce_weight', 0.2)
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.patience_counter = 0
    
    def _create_model(self):
        """Create the segmentation model"""
        model = smp.Unet(
            encoder_name=self.config.get('encoder_name', 'resnet34'),
            encoder_weights=self.config.get('encoder_weights', 'imagenet'),
            in_channels=self.config.get('in_channels', 3),
            classes=self.config.get('num_classes', 1),
            activation=None  # We'll apply sigmoid in loss function
        )
        
        logger.info(f"Created model: {self.config.get('encoder_name', 'resnet34')} U-Net")
        return model
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.config.get('optimizer', 'adam').lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        elif self.config.get('optimizer', 'adam').lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-2)
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-3),
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.get('scheduler', 'cosine').lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif self.config.get('scheduler', 'cosine').lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=self.config.get('min_lr', 1e-6)
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
    
    def calculate_metrics(self, pred, target, threshold=0.5):
        """Calculate segmentation metrics"""
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1).cpu().numpy()
        target_flat = target_binary.view(-1).cpu().numpy()
        
        # Calculate metrics
        intersection = (pred_binary * target_binary).sum().item()
        union = pred_binary.sum().item() + target_binary.sum().item() - intersection
        
        # IoU (Intersection over Union)
        iou = intersection / (union + 1e-6)
        
        # Dice coefficient
        dice = (2.0 * intersection) / (pred_binary.sum().item() + target_binary.sum().item() + 1e-6)
        
        # Pixel accuracy
        accuracy = accuracy_score(target_flat, pred_flat)
        
        # Precision and Recall (for defect pixels)
        precision = precision_score(target_flat, pred_flat, zero_division=0)
        recall = recall_score(target_flat, pred_flat, zero_division=0)
        f1 = f1_score(target_flat, pred_flat, zero_division=0)
        
        return {
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0}
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss, loss_components = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            metrics = self.calculate_metrics(outputs, masks)
            
            # Update running totals
            total_loss += loss.item()
            total_metrics['dice'] += metrics['dice']
            total_metrics['iou'] += metrics['iou']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['dice']:.4f}",
                'iou': f"{metrics['iou']:.4f}"
            })
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_metrics['dice'] / len(train_loader)
        avg_iou = total_metrics['iou'] / len(train_loader)
        
        return avg_loss, avg_dice, avg_iou
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss, _ = self.criterion(outputs, masks)
                
                # Calculate metrics
                metrics = self.calculate_metrics(outputs, masks)
                
                # Update running totals
                total_loss += loss.item()
                total_metrics['dice'] += metrics['dice']
                total_metrics['iou'] += metrics['iou']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{metrics['dice']:.4f}",
                    'iou': f"{metrics['iou']:.4f}"
                })
        
        # Calculate averages
        avg_loss = total_loss / len(val_loader)
        avg_dice = total_metrics['dice'] / len(val_loader)
        avg_iou = total_metrics['iou'] / len(val_loader)
        
        return avg_loss, avg_dice, avg_iou
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'config': self.config,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config['output_dir']) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config['output_dir']) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config.get('epochs', 100)):
            logger.info(f"Epoch {epoch + 1}/{self.config.get('epochs', 100)}")
            
            # Train
            train_loss, train_dice, train_iou = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            
            # Check for best model
            is_best = False
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.best_val_loss = val_loss
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_freq', 10) == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
            logger.info(f"Best Val Dice: {self.best_val_dice:.4f}")
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 20):
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint(epoch + 1, False)
        
        # Plot training history
        self.plot_training_history()
        
        logger.info("Training completed!")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice plot
        axes[0, 1].plot(self.history['train_dice'], label='Train Dice')
        axes[0, 1].plot(self.history['val_dice'], label='Val Dice')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU plot
        axes[1, 0].plot(self.history['train_iou'], label='Train IoU')
        axes[1, 0].plot(self.history['val_iou'], label='Val IoU')
        axes[1, 0].set_title('IoU (Intersection over Union)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        if hasattr(self.scheduler, 'get_last_lr'):
            lr_history = [self.scheduler.get_last_lr()[0] for _ in range(len(self.history['train_loss']))]
            axes[1, 1].plot(lr_history)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(self.config['output_dir']) / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

def get_default_training_config():
    """Get default training configuration"""
    return {
        # Model parameters
        'encoder_name': 'resnet34',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'num_classes': 1,
        
        # Training parameters
        'batch_size': 8,
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'min_lr': 1e-6,
        
        # Loss function parameters
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'dice_weight': 0.5,
        'focal_weight': 0.3,
        'bce_weight': 0.2,
        
        # Training control
        'patience': 20,
        'save_freq': 10,
        
        # Data augmentation
        'use_augmentation': True,
        
        # Paths (to be set by user)
        'dataset_dir': '',
        'output_dir': '',
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Defect Segmentation Model')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Directory containing train/val splits')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for model and logs')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to training configuration JSON file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_training_config()
    
    # Update paths
    config['dataset_dir'] = args.dataset_dir
    config['output_dir'] = args.output_dir
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(Path(args.output_dir) / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup data transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DefectSegmentationDataset(
        images_dir=Path(args.dataset_dir) / 'train' / 'images',
        masks_dir=Path(args.dataset_dir) / 'train' / 'masks',
        transform=train_transform,
        augment=config.get('use_augmentation', True)
    )
    
    val_dataset = DefectSegmentationDataset(
        images_dir=Path(args.dataset_dir) / 'val' / 'images',
        masks_dir=Path(args.dataset_dir) / 'val' / 'masks',
        transform=val_transform,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = SegmentationTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.best_val_dice = checkpoint['best_val_dice']
        trainer.history = checkpoint['history']
    
    # Start training
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()

