#!/usr/bin/env python3
"""
Advanced Defect Mask Generator for Chip and Check Detection
===========================================================

This script converts YOLO bounding box annotations into high-quality segmentation masks
using advanced computer vision techniques. It implements a multi-stage pipeline that:

1. Reduces noise and enhances defect features
2. Uses adaptive thresholding for robust segmentation
3. Applies morphological operations for mask refinement
4. Generates multi-channel feature-enhanced training data

The pipeline is designed to be robust and succeed regardless of image quality variations.

Author: Manus AI Agent
Date: 2025-01-14
"""

import os
import cv2
import numpy as np
import glob
from pathlib import Path
import logging
import json
from typing import Tuple, List, Dict, Optional
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mask_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedMaskGenerator:
    """
    Advanced mask generator using multi-stage computer vision pipeline
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the mask generator with configuration parameters
        
        Args:
            config: Dictionary containing processing parameters
        """
        self.config = config
        self.stats = {
            'processed': 0,
            'failed': 0,
            'total_defects': 0,
            'avg_defect_size': 0
        }
        
    def load_yolo_annotation(self, annotation_path: str, img_width: int, img_height: int) -> List[Tuple]:
        """
        Load YOLO format annotations and convert to pixel coordinates
        
        Args:
            annotation_path: Path to YOLO annotation file
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            List of tuples (class_id, x_min, y_min, x_max, y_max)
        """
        annotations = []
        
        if not os.path.exists(annotation_path):
            logger.warning(f"Annotation file not found: {annotation_path}")
            return annotations
            
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert from YOLO format (normalized) to pixel coordinates
                    x_min = int((x_center - width/2) * img_width)
                    y_min = int((y_center - height/2) * img_height)
                    x_max = int((x_center + width/2) * img_width)
                    y_max = int((y_center + height/2) * img_height)
                    
                    # Ensure coordinates are within image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width, x_max)
                    y_max = min(img_height, y_max)
                    
                    annotations.append((class_id, x_min, y_min, x_max, y_max))
                    
        except Exception as e:
            logger.error(f"Error reading annotation file {annotation_path}: {e}")
            
        return annotations
    
    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction and preprocessing to the cropped defect region
        
        Args:
            crop: Cropped image region containing the defect
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Noise reduction using median filter
        # Median filter is excellent for removing salt-and-pepper noise
        denoised = cv2.medianBlur(crop, self.config['median_kernel_size'])
        
        # Step 2: Additional Gaussian blur for smoothing
        if self.config['use_gaussian_blur']:
            denoised = cv2.GaussianBlur(denoised, 
                                      (self.config['gaussian_kernel_size'], self.config['gaussian_kernel_size']), 
                                      self.config['gaussian_sigma'])
        
        # Step 3: Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.config['use_clahe']:
            clahe = cv2.createCLAHE(clipLimit=self.config['clahe_clip_limit'], 
                                   tileGridSize=(self.config['clahe_tile_size'], self.config['clahe_tile_size']))
            denoised = clahe.apply(denoised)
        
        return denoised
    
    def enhance_defect_features(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance defect features using morphological operations
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Feature-enhanced image
        """
        # Create morphological kernel
        kernel_size = self.config['tophat_kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply Top-Hat transform to enhance bright defects on dark background
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Apply Black-Hat transform to enhance dark defects on bright background
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine both transformations
        enhanced = cv2.add(image, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
        
        return enhanced
    
    def adaptive_threshold_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for robust segmentation
        
        Args:
            image: Feature-enhanced grayscale image
            
        Returns:
            Binary mask
        """
        # Method 1: Adaptive Gaussian thresholding
        adaptive_gaussian = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config['adaptive_block_size'],
            self.config['adaptive_c_gaussian']
        )
        
        # Method 2: Adaptive Mean thresholding
        adaptive_mean = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.config['adaptive_block_size'],
            self.config['adaptive_c_mean']
        )
        
        # Method 3: Otsu's thresholding
        _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine all three methods using majority voting
        combined = np.zeros_like(image)
        vote_sum = adaptive_gaussian.astype(np.float32) + adaptive_mean.astype(np.float32) + otsu_thresh.astype(np.float32)
        combined[vote_sum >= (2 * 255)] = 255  # At least 2 out of 3 methods agree
        
        return combined.astype(np.uint8)
    
    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine the binary mask using morphological operations
        
        Args:
            mask: Binary mask from thresholding
            
        Returns:
            Refined binary mask
        """
        # Create kernels for morphological operations
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (self.config['opening_kernel_size'], self.config['opening_kernel_size']))
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (self.config['closing_kernel_size'], self.config['closing_kernel_size']))
        
        # Step 1: Opening to remove small noise
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
        
        # Step 2: Closing to fill holes in defects
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)
        
        # Step 3: Remove very small components (noise)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        # Filter out components smaller than minimum size
        min_area = self.config['min_defect_area']
        refined_mask = np.zeros_like(closed)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                refined_mask[labels == i] = 255
        
        return refined_mask
    
    def generate_edge_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate edge mask using Canny edge detection
        
        Args:
            image: Grayscale image
            
        Returns:
            Edge mask
        """
        # Apply Gaussian blur before edge detection
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Canny edge detection with automatic threshold calculation
        sigma = 0.33
        median_val = np.median(blurred)
        lower_thresh = int(max(0, (1.0 - sigma) * median_val))
        upper_thresh = int(min(255, (1.0 + sigma) * median_val))
        
        edges = cv2.Canny(blurred, lower_thresh, upper_thresh)
        
        # Dilate edges slightly to make them more prominent
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def process_single_defect(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single defect region and generate multi-channel output
        
        Args:
            image: Full input image
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            Tuple of (original_crop, attention_mask, edge_mask)
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Extract the crop
        if len(image.shape) == 3:
            crop = image[y_min:y_max, x_min:x_max]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = image[y_min:y_max, x_min:x_max]
            crop = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
        
        # Ensure minimum crop size
        if crop_gray.shape[0] < 10 or crop_gray.shape[1] < 10:
            logger.warning(f"Crop too small: {crop_gray.shape}, padding...")
            # Pad the crop to minimum size
            pad_h = max(0, 10 - crop_gray.shape[0])
            pad_w = max(0, 10 - crop_gray.shape[1])
            crop_gray = np.pad(crop_gray, ((0, pad_h), (0, pad_w)), mode='reflect')
            crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        try:
            # Step 1: Preprocess the crop
            preprocessed = self.preprocess_crop(crop_gray)
            
            # Step 2: Enhance defect features
            enhanced = self.enhance_defect_features(preprocessed)
            
            # Step 3: Generate attention mask
            attention_mask = self.adaptive_threshold_segmentation(enhanced)
            
            # Step 4: Refine the mask
            refined_mask = self.refine_mask(attention_mask)
            
            # Step 5: Generate edge mask
            edge_mask = self.generate_edge_mask(preprocessed)
            
            return crop, refined_mask, edge_mask
            
        except Exception as e:
            logger.error(f"Error processing defect crop: {e}")
            # Return fallback masks
            h, w = crop_gray.shape
            fallback_mask = np.zeros((h, w), dtype=np.uint8)
            return crop, fallback_mask, fallback_mask
    
    def process_image(self, image_path: str, annotation_path: str, output_dir: str) -> bool:
        """
        Process a single image and generate all output masks
        
        Args:
            image_path: Path to input image
            annotation_path: Path to YOLO annotation file
            output_dir: Output directory for generated masks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            h, w = image.shape[:2]
            
            # Load annotations
            annotations = self.load_yolo_annotation(annotation_path, w, h)
            if not annotations:
                logger.warning(f"No annotations found for {image_path}")
                return False
            
            # Get base filename
            base_name = Path(image_path).stem
            
            # Initialize full-size output masks
            full_attention_mask = np.zeros((h, w), dtype=np.uint8)
            full_edge_mask = np.zeros((h, w), dtype=np.uint8)
            full_original_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
            defect_count = 0
            total_defect_area = 0
            
            # Process each defect
            for class_id, x_min, y_min, x_max, y_max in annotations:
                # Only process chip (class 0) and check (class 1) defects
                if class_id not in [0, 1]:
                    continue
                
                # Process the defect
                crop, attention_mask, edge_mask = self.process_single_defect(image, (x_min, y_min, x_max, y_max))
                
                # Place masks back onto full-size canvas
                crop_h, crop_w = attention_mask.shape
                
                # Ensure we don't exceed image boundaries
                end_y = min(y_min + crop_h, h)
                end_x = min(x_min + crop_w, w)
                actual_h = end_y - y_min
                actual_w = end_x - x_min
                
                # Place attention mask
                full_attention_mask[y_min:end_y, x_min:end_x] = np.maximum(
                    full_attention_mask[y_min:end_y, x_min:end_x],
                    attention_mask[:actual_h, :actual_w]
                )
                
                # Place edge mask
                full_edge_mask[y_min:end_y, x_min:end_x] = np.maximum(
                    full_edge_mask[y_min:end_y, x_min:end_x],
                    edge_mask[:actual_h, :actual_w]
                )
                
                # Place original crop (for visualization)
                if len(crop.shape) == 3:
                    full_original_mask[y_min:end_y, x_min:end_x] = crop[:actual_h, :actual_w]
                
                defect_count += 1
                total_defect_area += np.sum(attention_mask > 0)
            
            # Save output masks
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_attention_mask.png"), full_attention_mask)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_edge_mask.png"), full_edge_mask)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_original_overlay.png"), full_original_mask)
            
            # Create and save combined visualization
            visualization = self.create_visualization(image, full_attention_mask, full_edge_mask)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_visualization.png"), visualization)
            
            # Update statistics
            self.stats['processed'] += 1
            self.stats['total_defects'] += defect_count
            if defect_count > 0:
                avg_area = total_defect_area / defect_count
                self.stats['avg_defect_size'] = (self.stats['avg_defect_size'] * (self.stats['processed'] - 1) + avg_area) / self.stats['processed']
            
            logger.info(f"Successfully processed {image_path}: {defect_count} defects found")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            self.stats['failed'] += 1
            return False
    
    def create_visualization(self, original: np.ndarray, attention_mask: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
        """
        Create a visualization combining original image with generated masks
        
        Args:
            original: Original input image
            attention_mask: Generated attention mask
            edge_mask: Generated edge mask
            
        Returns:
            Combined visualization image
        """
        # Convert to RGB if needed
        if len(original.shape) == 3:
            vis_img = original.copy()
        else:
            vis_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Create colored overlays
        attention_overlay = np.zeros_like(vis_img)
        attention_overlay[:, :, 2] = attention_mask  # Red channel for attention
        
        edge_overlay = np.zeros_like(vis_img)
        edge_overlay[:, :, 1] = edge_mask  # Green channel for edges
        
        # Combine overlays with original image
        alpha = 0.3
        visualization = cv2.addWeighted(vis_img, 1.0, attention_overlay, alpha, 0)
        visualization = cv2.addWeighted(visualization, 1.0, edge_overlay, alpha, 0)
        
        return visualization

def get_default_config() -> Dict:
    """
    Get default configuration parameters for the mask generator
    
    Returns:
        Dictionary with default configuration
    """
    return {
        # Preprocessing parameters
        'median_kernel_size': 5,
        'use_gaussian_blur': True,
        'gaussian_kernel_size': 3,
        'gaussian_sigma': 1.0,
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': 8,
        
        # Feature enhancement parameters
        'tophat_kernel_size': 7,
        
        # Adaptive thresholding parameters
        'adaptive_block_size': 11,
        'adaptive_c_gaussian': 2,
        'adaptive_c_mean': 2,
        
        # Morphological refinement parameters
        'opening_kernel_size': 3,
        'closing_kernel_size': 5,
        'min_defect_area': 10,
    }

def main():
    """
    Main function to run the mask generation pipeline
    """
    parser = argparse.ArgumentParser(description='Advanced Defect Mask Generator')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing images and YOLO annotations')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for generated masks (default: input_dir/generated_masks)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration JSON file (optional)')
    
    args = parser.parse_args()
    
    # Set up paths
    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'generated_masks'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Save configuration for reference
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize mask generator
    generator = AdvancedMaskGenerator(config)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(str(input_dir / ext)))
        image_files.extend(glob.glob(str(input_dir / ext.upper())))
    
    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} image files")
    
    # Process all images
    successful = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        # Find corresponding annotation file
        base_name = Path(image_path).stem
        annotation_path = input_dir / f"{base_name}.txt"
        
        if generator.process_image(image_path, str(annotation_path), str(output_dir)):
            successful += 1
    
    # Print final statistics
    logger.info("=" * 50)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total images processed: {generator.stats['processed']}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {generator.stats['failed']}")
    logger.info(f"Total defects found: {generator.stats['total_defects']}")
    logger.info(f"Average defect size: {generator.stats['avg_defect_size']:.2f} pixels")
    logger.info(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()

