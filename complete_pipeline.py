#!/usr/bin/env python3
"""
Complete End-to-End Defect Detection Pipeline
============================================

This script provides a complete pipeline from YOLO dataset to trained segmentation model:
1. Process YOLO dataset with train/test/val splits
2. Generate high-quality segmentation masks
3. Prepare dataset for training
4. Train state-of-the-art segmentation model
5. Evaluate and validate results

Usage:
    python complete_pipeline.py --yolo_dataset_path "D:\Photomask\yolodataset\ev dataset"

Author: Manus AI Agent
Date: 2025-01-14
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

# Import our custom modules
from defect_mask_generator import AdvancedMaskGenerator, get_default_config
from dataset_utils import YOLODatasetProcessor, SegmentationDatasetCreator, convert_windows_path
from train_segmentation_model import main as train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompletePipeline:
    """
    Complete pipeline for defect detection from YOLO to trained model
    """
    
    def __init__(self, yolo_dataset_path: str, output_base_dir: str = None):
        """
        Initialize the complete pipeline
        
        Args:
            yolo_dataset_path: Path to YOLO dataset (contains images/ and labels/ folders)
            output_base_dir: Base directory for all outputs (default: yolo_dataset_path/pipeline_output)
        """
        # Convert Windows path format
        self.yolo_dataset_path = Path(convert_windows_path(yolo_dataset_path))
        
        if output_base_dir:
            self.output_base_dir = Path(output_base_dir)
        else:
            self.output_base_dir = self.yolo_dataset_path / 'pipeline_output'
        
        # Create output directories
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.masks_output_dir = self.output_base_dir / 'generated_masks'
        self.segmentation_dataset_dir = self.output_base_dir / 'segmentation_dataset'
        self.model_output_dir = self.output_base_dir / 'trained_model'
        
        # Validate YOLO dataset structure
        self._validate_yolo_structure()
        
        # Initialize components
        self.mask_config = get_default_config()
        self.mask_generator = AdvancedMaskGenerator(self.mask_config)
        
        logger.info(f"Pipeline initialized:")
        logger.info(f"  YOLO dataset: {self.yolo_dataset_path}")
        logger.info(f"  Output directory: {self.output_base_dir}")
    
    def _validate_yolo_structure(self):
        """Validate that the YOLO dataset has the correct structure"""
        required_dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        
        for dir_path in required_dirs:
            full_path = self.yolo_dataset_path / dir_path
            if not full_path.exists():
                logger.warning(f"Directory not found: {full_path}")
            else:
                logger.info(f"Found: {full_path}")
    
    def analyze_dataset(self) -> Dict:
        """
        Analyze the YOLO dataset and collect statistics
        
        Returns:
            Dictionary with dataset analysis
        """
        logger.info("=" * 60)
        logger.info("STEP 1: ANALYZING YOLO DATASET")
        logger.info("=" * 60)
        
        analysis = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        for split in ['train', 'val', 'test']:
            logger.info(f"\nAnalyzing {split} split...")
            
            images_dir = self.yolo_dataset_path / 'images' / split
            labels_dir = self.yolo_dataset_path / 'labels' / split
            
            if not images_dir.exists() or not labels_dir.exists():
                logger.warning(f"Skipping {split} split - directories not found")
                continue
            
            # Create a temporary processor for this split
            split_processor = YOLODatasetProcessor(str(self.yolo_dataset_path))
            split_processor.images_dir = images_dir
            split_processor.labels_dir = labels_dir
            
            # Scan this split
            stats = split_processor.scan_dataset()
            analysis[split] = stats
            
            logger.info(f"  {split.upper()} SPLIT STATISTICS:")
            logger.info(f"    Total images: {stats['total_images']}")
            logger.info(f"    Images with labels: {stats['images_with_labels']}")
            logger.info(f"    Empty labels: {stats['empty_labels']}")
            logger.info(f"    Total annotations: {stats['total_annotations']}")
            logger.info(f"    Chip defects: {stats['chip_defects']}")
            logger.info(f"    Check defects: {stats['check_defects']}")
        
        # Save analysis
        with open(self.output_base_dir / 'dataset_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def generate_masks_for_split(self, split: str) -> bool:
        """
        Generate segmentation masks for a specific split
        
        Args:
            split: Split name ('train', 'val', or 'test')
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"\nGenerating masks for {split} split...")
        
        images_dir = self.yolo_dataset_path / 'images' / split
        labels_dir = self.yolo_dataset_path / 'labels' / split
        output_dir = self.masks_output_dir / split
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"Skipping {split} split - directories not found")
            return False
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(ext)))
            image_files.extend(list(images_dir.glob(ext.upper())))
        
        if not image_files:
            logger.warning(f"No image files found in {images_dir}")
            return False
        
        logger.info(f"Processing {len(image_files)} images in {split} split...")
        
        successful = 0
        for image_path in image_files:
            # Find corresponding annotation file
            base_name = image_path.stem
            annotation_path = labels_dir / f"{base_name}.txt"
            
            if self.mask_generator.process_image(str(image_path), str(annotation_path), str(output_dir)):
                successful += 1
        
        logger.info(f"Successfully processed {successful}/{len(image_files)} images in {split} split")
        return successful > 0
    
    def generate_all_masks(self):
        """Generate segmentation masks for all splits"""
        logger.info("=" * 60)
        logger.info("STEP 2: GENERATING SEGMENTATION MASKS")
        logger.info("=" * 60)
        
        # Save mask generation config
        with open(self.masks_output_dir / 'mask_config.json', 'w') as f:
            json.dump(self.mask_config, f, indent=2)
        
        # Process each split
        for split in ['train', 'val', 'test']:
            self.generate_masks_for_split(split)
        
        # Print final statistics
        logger.info("\nMask generation complete!")
        logger.info(f"Generated masks saved to: {self.masks_output_dir}")
    
    def prepare_segmentation_dataset(self):
        """Prepare the dataset for segmentation training"""
        logger.info("=" * 60)
        logger.info("STEP 3: PREPARING SEGMENTATION DATASET")
        logger.info("=" * 60)
        
        # Create segmentation dataset structure
        self.segmentation_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val']:
            # Create directories
            (self.segmentation_dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.segmentation_dataset_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
            
            # Copy images and masks
            source_images_dir = self.yolo_dataset_path / 'images' / split
            source_masks_dir = self.masks_output_dir / split
            
            target_images_dir = self.segmentation_dataset_dir / split / 'images'
            target_masks_dir = self.segmentation_dataset_dir / split / 'masks'
            
            if source_images_dir.exists() and source_masks_dir.exists():
                logger.info(f"Preparing {split} split...")
                
                # Copy images
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(list(source_images_dir.glob(ext)))
                    image_files.extend(list(source_images_dir.glob(ext.upper())))
                
                copied_count = 0
                for image_file in image_files:
                    # Check if corresponding mask exists
                    mask_file = source_masks_dir / f"{image_file.stem}_attention_mask.png"
                    if mask_file.exists():
                        # Copy image
                        target_image = target_images_dir / f"{image_file.stem}.png"
                        shutil.copy2(image_file, target_image)
                        
                        # Copy mask
                        target_mask = target_masks_dir / f"{image_file.stem}.png"
                        shutil.copy2(mask_file, target_mask)
                        
                        copied_count += 1
                
                logger.info(f"  Copied {copied_count} image-mask pairs for {split}")
        
        # Create dataset info
        dataset_info = {
            'name': 'Chip and Check Defect Segmentation Dataset',
            'description': 'Generated from YOLO annotations using advanced computer vision pipeline',
            'classes': ['background', 'defect'],
            'class_names': ['background', 'chip_or_check_defect'],
            'source_dataset': str(self.yolo_dataset_path),
            'generation_config': self.mask_config
        }
        
        with open(self.segmentation_dataset_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Segmentation dataset prepared: {self.segmentation_dataset_dir}")
    
    def train_segmentation_model(self, config_overrides: Dict = None):
        """Train the segmentation model"""
        logger.info("=" * 60)
        logger.info("STEP 4: TRAINING SEGMENTATION MODEL")
        logger.info("=" * 60)
        
        # Create model output directory
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare training configuration
        from train_segmentation_model import get_default_training_config
        training_config = get_default_training_config()
        
        # Update with user overrides
        if config_overrides:
            training_config.update(config_overrides)
        
        # Set paths
        training_config['dataset_dir'] = str(self.segmentation_dataset_dir)
        training_config['output_dir'] = str(self.model_output_dir)
        
        # Save training config
        with open(self.model_output_dir / 'training_config.json', 'w') as f:
            json.dump(training_config, f, indent=2)
        
        # Import and run training
        try:
            import sys
            from train_segmentation_model import main as train_main
            
            # Temporarily modify sys.argv to pass arguments to training script
            original_argv = sys.argv.copy()
            sys.argv = [
                'train_segmentation_model.py',
                '--dataset_dir', str(self.segmentation_dataset_dir),
                '--output_dir', str(self.model_output_dir),
                '--config_file', str(self.model_output_dir / 'training_config.json')
            ]
            
            # Run training
            train_main()
            
            # Restore original argv
            sys.argv = original_argv
            
            logger.info("Model training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            logger.info("You can manually run training with:")
            logger.info(f"python train_segmentation_model.py --dataset_dir {self.segmentation_dataset_dir} --output_dir {self.model_output_dir}")
    
    def create_inference_script(self):
        """Create an inference script for the trained model"""
        inference_script = '''#!/usr/bin/env python3
"""
Inference Script for Trained Defect Segmentation Model
=====================================================

This script loads the trained model and performs inference on new images.

Usage:
    python inference.py --model_path best_model.pth --image_path test_image.jpg
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import segmentation_models_pytorch as smp
from torchvision import transforms

def load_model(model_path, device):
    """Load the trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = smp.Unet(
        encoder_name=config.get('encoder_name', 'resnet34'),
        encoder_weights=None,  # Don't load pretrained weights
        in_channels=config.get('in_channels', 3),
        classes=config.get('num_classes', 1),
        activation=None
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path):
    """Preprocess image for inference"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def postprocess_prediction(prediction, original_shape, threshold=0.5):
    """Postprocess model prediction"""
    # Apply sigmoid and threshold
    prediction = torch.sigmoid(prediction)
    mask = (prediction > threshold).float()
    
    # Convert to numpy
    mask = mask.squeeze().cpu().numpy()
    
    # Resize to original shape
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
    
    return (mask * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Defect Segmentation Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save output')
    parser.add_argument('--threshold', type=float, default=0.5, help='Segmentation threshold')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, device)
    
    # Load and preprocess image
    print("Processing image...")
    image_tensor, original_image = preprocess_image(args.image_path)
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Postprocess
    mask = postprocess_prediction(prediction, original_image.shape[:2], args.threshold)
    
    # Create visualization
    visualization = original_image.copy()
    mask_colored = np.zeros_like(original_image)
    mask_colored[:, :, 0] = mask  # Red channel for defects
    
    # Overlay mask on original image
    alpha = 0.3
    visualization = cv2.addWeighted(visualization, 1.0, mask_colored, alpha, 0)
    
    # Save results
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        input_path = Path(args.image_path)
        output_path = input_path.parent / f"{input_path.stem}_segmented.png"
    
    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_path.parent / f"{output_path.stem}_mask.png"), mask)
    
    print(f"Results saved to: {output_path}")
    print(f"Mask saved to: {output_path.parent / f'{output_path.stem}_mask.png'}")

if __name__ == "__main__":
    main()
'''
        
        # Save inference script
        inference_path = self.model_output_dir / 'inference.py'
        with open(inference_path, 'w') as f:
            f.write(inference_script)
        
        logger.info(f"Inference script created: {inference_path}")
    
    def run_complete_pipeline(self, training_config_overrides: Dict = None):
        """Run the complete pipeline from start to finish"""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE DEFECT DETECTION PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Step 1: Analyze dataset
            analysis = self.analyze_dataset()
            
            # Step 2: Generate masks
            self.generate_all_masks()
            
            # Step 3: Prepare segmentation dataset
            self.prepare_segmentation_dataset()
            
            # Step 4: Train model
            self.train_segmentation_model(training_config_overrides)
            
            # Step 5: Create inference script
            self.create_inference_script()
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"All outputs saved to: {self.output_base_dir}")
            logger.info(f"Trained model: {self.model_output_dir / 'best_model.pth'}")
            logger.info(f"Inference script: {self.model_output_dir / 'inference.py'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Complete Defect Detection Pipeline')
    parser.add_argument('--yolo_dataset_path', type=str, required=True,
                       help='Path to YOLO dataset (e.g., "D:\\Photomask\\yolodataset\\ev dataset")')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for all results (default: dataset_path/pipeline_output)')
    parser.add_argument('--training_config', type=str, default=None,
                       help='Path to training configuration JSON file')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip model training (only generate masks and prepare dataset)')
    
    args = parser.parse_args()
    
    # Load training config overrides if provided
    training_overrides = None
    if args.training_config and os.path.exists(args.training_config):
        with open(args.training_config, 'r') as f:
            training_overrides = json.load(f)
    
    # Initialize pipeline
    pipeline = CompletePipeline(args.yolo_dataset_path, args.output_dir)
    
    if args.skip_training:
        # Run only data preparation steps
        pipeline.analyze_dataset()
        pipeline.generate_all_masks()
        pipeline.prepare_segmentation_dataset()
        pipeline.create_inference_script()
    else:
        # Run complete pipeline
        pipeline.run_complete_pipeline(training_overrides)

if __name__ == "__main__":
    main()

