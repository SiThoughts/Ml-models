#!/usr/bin/env python3
"""
Dataset Utilities for YOLO to Segmentation Conversion
====================================================

This module provides utility functions for handling YOLO datasets,
converting them to segmentation format, and preparing them for training.

Author: Manus AI Agent
Date: 2025-01-14
"""

import os
import cv2
import numpy as np
import glob
import json
import shutil
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import random

logger = logging.getLogger(__name__)

class YOLODatasetProcessor:
    """
    Processes YOLO datasets and converts them to segmentation format
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset processor
        
        Args:
            dataset_path: Path to YOLO dataset (should contain 'images' and 'labels' folders)
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / 'images'
        self.labels_dir = self.dataset_path / 'labels'
        
        # Validate dataset structure
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")
        
        self.stats = {
            'total_images': 0,
            'images_with_labels': 0,
            'empty_labels': 0,
            'total_annotations': 0,
            'chip_defects': 0,
            'check_defects': 0
        }
    
    def scan_dataset(self) -> Dict:
        """
        Scan the dataset and collect statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Scanning dataset...")
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(self.images_dir / ext)))
            image_files.extend(glob.glob(str(self.images_dir / ext.upper())))
        
        self.stats['total_images'] = len(image_files)
        
        # Check corresponding label files
        for image_path in image_files:
            base_name = Path(image_path).stem
            label_path = self.labels_dir / f"{base_name}.txt"
            
            if label_path.exists():
                self.stats['images_with_labels'] += 1
                
                # Count annotations in this file
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    if not lines or all(line.strip() == '' for line in lines):
                        self.stats['empty_labels'] += 1
                    else:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                self.stats['total_annotations'] += 1
                                
                                if class_id == 0:  # chip defect
                                    self.stats['chip_defects'] += 1
                                elif class_id == 1:  # check defect
                                    self.stats['check_defects'] += 1
                                    
                except Exception as e:
                    logger.warning(f"Error reading label file {label_path}: {e}")
        
        logger.info("Dataset scan complete:")
        logger.info(f"  Total images: {self.stats['total_images']}")
        logger.info(f"  Images with labels: {self.stats['images_with_labels']}")
        logger.info(f"  Empty label files: {self.stats['empty_labels']}")
        logger.info(f"  Total annotations: {self.stats['total_annotations']}")
        logger.info(f"  Chip defects: {self.stats['chip_defects']}")
        logger.info(f"  Check defects: {self.stats['check_defects']}")
        
        return self.stats
    
    def get_annotated_images(self) -> List[Tuple[str, str]]:
        """
        Get list of images that have non-empty annotations
        
        Returns:
            List of tuples (image_path, label_path) for images with annotations
        """
        annotated_pairs = []
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(self.images_dir / ext)))
            image_files.extend(glob.glob(str(self.images_dir / ext.upper())))
        
        for image_path in image_files:
            base_name = Path(image_path).stem
            label_path = self.labels_dir / f"{base_name}.txt"
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Check if file has valid annotations
                    has_annotations = False
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id in [0, 1]:  # chip or check defect
                                has_annotations = True
                                break
                    
                    if has_annotations:
                        annotated_pairs.append((image_path, str(label_path)))
                        
                except Exception as e:
                    logger.warning(f"Error reading label file {label_path}: {e}")
        
        logger.info(f"Found {len(annotated_pairs)} images with valid annotations")
        return annotated_pairs
    
    def create_train_val_split(self, annotated_pairs: List[Tuple[str, str]], 
                              val_split: float = 0.2, 
                              random_seed: int = 42) -> Tuple[List, List]:
        """
        Split annotated images into training and validation sets
        
        Args:
            annotated_pairs: List of (image_path, label_path) tuples
            val_split: Fraction of data to use for validation
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_pairs, val_pairs)
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        train_pairs, val_pairs = train_test_split(
            annotated_pairs, 
            test_size=val_split, 
            random_state=random_seed,
            shuffle=True
        )
        
        logger.info(f"Dataset split: {len(train_pairs)} training, {len(val_pairs)} validation")
        return train_pairs, val_pairs

class SegmentationDatasetCreator:
    """
    Creates segmentation dataset from processed YOLO data
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the dataset creator
        
        Args:
            output_dir: Directory to create the segmentation dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.train_images_dir = self.output_dir / 'train' / 'images'
        self.train_masks_dir = self.output_dir / 'train' / 'masks'
        self.val_images_dir = self.output_dir / 'val' / 'images'
        self.val_masks_dir = self.output_dir / 'val' / 'masks'
        
        for dir_path in [self.train_images_dir, self.train_masks_dir, 
                        self.val_images_dir, self.val_masks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def copy_and_process_data(self, train_pairs: List[Tuple[str, str]], 
                             val_pairs: List[Tuple[str, str]],
                             mask_generator) -> None:
        """
        Copy images and generate masks for training and validation sets
        
        Args:
            train_pairs: List of training (image_path, label_path) pairs
            val_pairs: List of validation (image_path, label_path) pairs
            mask_generator: Instance of AdvancedMaskGenerator
        """
        logger.info("Creating training dataset...")
        self._process_split(train_pairs, self.train_images_dir, self.train_masks_dir, mask_generator)
        
        logger.info("Creating validation dataset...")
        self._process_split(val_pairs, self.val_images_dir, self.val_masks_dir, mask_generator)
        
        # Create dataset info file
        dataset_info = {
            'train_samples': len(train_pairs),
            'val_samples': len(val_pairs),
            'classes': ['background', 'defect'],
            'class_names': ['background', 'chip_or_check_defect'],
            'created_by': 'Advanced Defect Mask Generator'
        }
        
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def _process_split(self, pairs: List[Tuple[str, str]], 
                      images_dir: Path, masks_dir: Path,
                      mask_generator) -> None:
        """
        Process a single split (train or val)
        
        Args:
            pairs: List of (image_path, label_path) pairs
            images_dir: Directory to save images
            masks_dir: Directory to save masks
            mask_generator: Instance of AdvancedMaskGenerator
        """
        from tqdm import tqdm
        
        for image_path, label_path in tqdm(pairs, desc=f"Processing {images_dir.parent.name}"):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                base_name = Path(image_path).stem
                h, w = image.shape[:2]
                
                # Load annotations
                annotations = mask_generator.load_yolo_annotation(label_path, w, h)
                
                if not annotations:
                    logger.warning(f"No valid annotations for {image_path}")
                    continue
                
                # Generate mask
                full_mask = np.zeros((h, w), dtype=np.uint8)
                
                for class_id, x_min, y_min, x_max, y_max in annotations:
                    if class_id not in [0, 1]:  # Only process chip and check defects
                        continue
                    
                    # Process the defect
                    crop, attention_mask, edge_mask = mask_generator.process_single_defect(
                        image, (x_min, y_min, x_max, y_max)
                    )
                    
                    # Place mask back onto full-size canvas
                    crop_h, crop_w = attention_mask.shape
                    end_y = min(y_min + crop_h, h)
                    end_x = min(x_min + crop_w, w)
                    actual_h = end_y - y_min
                    actual_w = end_x - x_min
                    
                    full_mask[y_min:end_y, x_min:end_x] = np.maximum(
                        full_mask[y_min:end_y, x_min:end_x],
                        attention_mask[:actual_h, :actual_w]
                    )
                
                # Save image and mask
                cv2.imwrite(str(images_dir / f"{base_name}.png"), image)
                cv2.imwrite(str(masks_dir / f"{base_name}.png"), full_mask)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")

def validate_dataset_structure(dataset_path: str) -> bool:
    """
    Validate that the dataset has the correct structure
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    dataset_path = Path(dataset_path)
    
    required_dirs = ['images', 'labels']
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            logger.error(f"Required directory not found: {dir_path}")
            return False
        if not dir_path.is_dir():
            logger.error(f"Path is not a directory: {dir_path}")
            return False
    
    return True

def convert_windows_path(path_str: str) -> str:
    """
    Convert Windows path format to cross-platform format
    
    Args:
        path_str: Windows path string (e.g., "D:Photomask> yolodataset >ev dataset")
        
    Returns:
        Normalized path string
    """
    # Handle the specific format from user
    if '>' in path_str:
        # Split by '>' and clean up spaces
        parts = [part.strip() for part in path_str.split('>')]
        # Join with proper path separators
        normalized = os.path.join(*parts)
    else:
        normalized = path_str
    
    # Convert to Path object for cross-platform compatibility
    return str(Path(normalized))

def create_class_mapping() -> Dict[int, str]:
    """
    Create mapping from class IDs to class names
    
    Returns:
        Dictionary mapping class IDs to names
    """
    return {
        0: 'chip_defect',
        1: 'check_defect'
    }

def analyze_defect_distribution(annotated_pairs: List[Tuple[str, str]]) -> Dict:
    """
    Analyze the distribution of defect types and sizes
    
    Args:
        annotated_pairs: List of (image_path, label_path) pairs
        
    Returns:
        Dictionary with distribution statistics
    """
    stats = {
        'chip_defects': 0,
        'check_defects': 0,
        'defect_sizes': [],
        'images_per_defect_count': {}
    }
    
    for image_path, label_path in annotated_pairs:
        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                continue
            h, w = image.shape[:2]
            
            # Load annotations
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            defect_count = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id in [0, 1]:
                        defect_count += 1
                        
                        if class_id == 0:
                            stats['chip_defects'] += 1
                        else:
                            stats['check_defects'] += 1
                        
                        # Calculate defect size
                        width = float(parts[3]) * w
                        height = float(parts[4]) * h
                        area = width * height
                        stats['defect_sizes'].append(area)
            
            # Track images by defect count
            if defect_count not in stats['images_per_defect_count']:
                stats['images_per_defect_count'][defect_count] = 0
            stats['images_per_defect_count'][defect_count] += 1
            
        except Exception as e:
            logger.warning(f"Error analyzing {image_path}: {e}")
    
    # Calculate statistics
    if stats['defect_sizes']:
        stats['avg_defect_size'] = np.mean(stats['defect_sizes'])
        stats['median_defect_size'] = np.median(stats['defect_sizes'])
        stats['min_defect_size'] = np.min(stats['defect_sizes'])
        stats['max_defect_size'] = np.max(stats['defect_sizes'])
    
    return stats

