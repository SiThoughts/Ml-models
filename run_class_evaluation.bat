@echo off
REM Class-Aware Model Evaluation Batch Script
REM ==========================================
REM This batch file runs the class-aware evaluation script with your specific paths.
REM Make sure to update the paths below to match your actual directory structure.

echo Starting class-aware model evaluation...
echo.

REM Update these paths to match your actual directory structure:
set MODEL_PATH="D:\Photomask\defect_detection_datasets_fixed\EV_dataset\pipeline_output\trained_model\best_model.pth"
set DATASET_DIR="D:\Photomask\defect_detection_datasets_fixed\EV_dataset\pipeline_output\segmentation_dataset"
set YOLO_LABELS_DIR="D:\Photomask\defect_detection_datasets_fixed\EV_dataset\labels\val"
set OUTPUT_DIR="D:\Photomask\defect_detection_datasets_fixed\EV_dataset\pipeline_output\class_evaluation_results"

echo Model Path: %MODEL_PATH%
echo Dataset Directory: %DATASET_DIR%
echo YOLO Labels Directory: %YOLO_LABELS_DIR%
echo Output Directory: %OUTPUT_DIR%
echo.

REM Run the evaluation script
python evaluate_by_class.py --model_path %MODEL_PATH% --dataset_dir %DATASET_DIR% --yolo_labels_dir %YOLO_LABELS_DIR% --output_dir %OUTPUT_DIR%

echo.
echo Evaluation complete! Check the output directory for results.
pause

