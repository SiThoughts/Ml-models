#!/usr/bin/env python3
"""
Test Script for Defect Detection Pipeline
========================================

This script performs basic validation tests on the pipeline components
to ensure everything is working correctly before running on real data.

Author: Manus AI Agent
Date: 2025-01-14
"""

import os
import sys
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging

# Import our modules
from defect_mask_generator import AdvancedMaskGenerator, get_default_config
from dataset_utils import YOLODatasetProcessor, convert_windows_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(width=512, height=512):
    """Create a synthetic test image with a defect"""
    # Create a base image (simulating a chip surface)
    image = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Add some texture
    noise = np.random.normal(0, 10, (height, width, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Add a synthetic defect (dark spot)
    center_x, center_y = width // 2, height // 2
    defect_size = 30
    cv2.circle(image, (center_x, center_y), defect_size, (50, 50, 50), -1)
    
    # Add some noise around the defect
    cv2.circle(image, (center_x + 50, center_y + 50), 10, (80, 80, 80), -1)
    
    return image

def create_test_annotation(width=512, height=512):
    """Create a test YOLO annotation"""
    # Defect at center, normalized coordinates
    center_x_norm = 0.5
    center_y_norm = 0.5
    width_norm = 0.15  # 15% of image width
    height_norm = 0.15  # 15% of image height
    
    # YOLO format: class_id x_center y_center width height
    annotation = f"0 {center_x_norm} {center_y_norm} {width_norm} {height_norm}\n"
    return annotation

def test_mask_generator():
    """Test the mask generation pipeline"""
    logger.info("Testing mask generation pipeline...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test image and annotation
        test_image = create_test_image()
        test_annotation = create_test_annotation()
        
        # Save test files
        image_path = temp_path / "test_image.png"
        annotation_path = temp_path / "test_image.txt"
        
        cv2.imwrite(str(image_path), test_image)
        with open(annotation_path, 'w') as f:
            f.write(test_annotation)
        
        # Test mask generator
        config = get_default_config()
        generator = AdvancedMaskGenerator(config)
        
        # Process the test image
        success = generator.process_image(
            str(image_path), 
            str(annotation_path), 
            str(temp_path)
        )
        
        if success:
            # Check if output files were created
            expected_files = [
                "test_image_attention_mask.png",
                "test_image_edge_mask.png",
                "test_image_original_overlay.png",
                "test_image_visualization.png"
            ]
            
            all_files_exist = True
            for filename in expected_files:
                file_path = temp_path / filename
                if not file_path.exists():
                    logger.error(f"Expected output file not found: {filename}")
                    all_files_exist = False
                else:
                    # Check if file is not empty
                    if file_path.stat().st_size == 0:
                        logger.error(f"Output file is empty: {filename}")
                        all_files_exist = False
            
            if all_files_exist:
                logger.info("✅ Mask generation test PASSED")
                return True
            else:
                logger.error("❌ Mask generation test FAILED - Missing output files")
                return False
        else:
            logger.error("❌ Mask generation test FAILED - Processing failed")
            return False

def test_dataset_utils():
    """Test dataset utility functions"""
    logger.info("Testing dataset utilities...")
    
    try:
        # Test Windows path conversion
        test_path = "D:Photomask> yolodataset >ev dataset"
        converted = convert_windows_path(test_path)
        logger.info(f"Path conversion: {test_path} -> {converted}")
        
        # Test with a mock dataset structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock dataset structure
            images_dir = temp_path / "images"
            labels_dir = temp_path / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            
            # Create a few test files
            for i in range(3):
                # Create image
                test_image = create_test_image()
                image_path = images_dir / f"test_{i}.png"
                cv2.imwrite(str(image_path), test_image)
                
                # Create annotation
                annotation_path = labels_dir / f"test_{i}.txt"
                with open(annotation_path, 'w') as f:
                    if i < 2:  # First two have annotations
                        f.write(create_test_annotation())
                    # Third one is empty (simulating empty labels)
            
            # Test dataset processor
            processor = YOLODatasetProcessor(str(temp_path))
            stats = processor.scan_dataset()
            
            # Validate results
            if stats['total_images'] == 3 and stats['images_with_labels'] == 3:
                logger.info("✅ Dataset utilities test PASSED")
                return True
            else:
                logger.error(f"❌ Dataset utilities test FAILED - Unexpected stats: {stats}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Dataset utilities test FAILED - Exception: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing module imports...")
    
    required_modules = [
        'cv2',
        'numpy',
        'torch',
        'torchvision',
        'sklearn',
        'matplotlib',
        'tqdm',
        'albumentations',
        'segmentation_models_pytorch'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module}: {e}")
            missing_modules.append(module)
    
    if not missing_modules:
        logger.info("✅ All module imports test PASSED")
        return True
    else:
        logger.error(f"❌ Module imports test FAILED - Missing: {missing_modules}")
        logger.error("Please install missing modules with: pip install -r requirements.txt")
        return False

def test_opencv_functionality():
    """Test OpenCV functionality with sample operations"""
    logger.info("Testing OpenCV functionality...")
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Test basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        
        # Test morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        
        # Test thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Test edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        logger.info("✅ OpenCV functionality test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenCV functionality test FAILED: {e}")
        return False

def run_all_tests():
    """Run all validation tests"""
    logger.info("=" * 60)
    logger.info("RUNNING PIPELINE VALIDATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("OpenCV Functionality", test_opencv_functionality),
        ("Dataset Utilities", test_dataset_utils),
        ("Mask Generation", test_mask_generator),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        icon = "✅" if result else "❌"
        logger.info(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests PASSED! Pipeline is ready to use.")
        return True
    else:
        logger.error("⚠️  Some tests FAILED. Please check the issues above.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("NEXT STEPS")
        logger.info("=" * 60)
        logger.info("1. Run the complete pipeline:")
        logger.info('   python complete_pipeline.py --yolo_dataset_path "D:\\Photomask\\yolodataset\\ev dataset"')
        logger.info("\n2. Or run individual components:")
        logger.info('   python defect_mask_generator.py --input_dir "path/to/images" --output_dir "path/to/output"')
        logger.info("\n3. Check the README.md for detailed usage instructions")
        
        return 0
    else:
        logger.error("\nPlease fix the failing tests before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

