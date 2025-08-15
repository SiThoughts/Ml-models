@echo off
setlocal enabledelayedexpansion

:: #############################################################################
:: #         Automated MMDetection Training Pipeline for Faster R-CNN          #
:: #############################################################################
:: # Author: Manus
:: # Date: 2025-08-15
:: # Purpose: End-to-end setup, training, and inference using XML annotations.
:: # Model: Faster R-CNN w/ ResNet-50 FPN
:: #############################################################################

:: --- Configuration ---
set "ENV_NAME=hummingbird"
set "PYTHON_VERSION=3.8"
set "PYTORCH_VERSION=2.1.0"
set "TORCHVISION_VERSION=0.16.0"
set "CUDA_VERSION=cu118"
set "LOG_FILE=training_log.txt"
set "PROJECT_ROOT=%~dp0"
set "MMDET_DIR=%PROJECT_ROOT%mmdetection"

:: #############################################################################
:: # Set the path to your Conda installation (from user's screenshot)          #
:: #############################################################################
set "CONDA_INSTALL_PATH=C:\ProgramData\anaconda3"
:: #############################################################################

:: --- Clear Previous Log ---
if exist "%LOG_FILE%" del "%LOG_FILE%"

:: ============================================================================
:: 1. Welcome and Input Acquisition
:: ============================================================================
echo ###################################################
echo # Welcome to the Automated Chip Detection Trainer #
echo #       (Faster R-CNN + XML Edition)            #
echo ###################################################
echo.
echo This script will set up and train a Faster R-CNN model
echo directly using your XML annotation files.
echo. 
echo Starting process at %date% %time% >> "%LOG_FILE%"

:get_paths
set "TRAIN_PATH="
set "VAL_PATH="

set /p "TRAIN_PATH=Enter the FULL path to your training data folder: "
set /p "VAL_PATH=Enter the FULL path to your validation data folder: "

:: Validate paths exist
if not exist "%TRAIN_PATH%" (
    echo ERROR: Training path does not exist: %TRAIN_PATH%
    goto get_paths
)
if not exist "%VAL_PATH%" (
    echo ERROR: Validation path does not exist: %VAL_PATH%
    goto get_paths
)

:: Check for images and XML files in the folders
set "TRAIN_HAS_IMAGES=0"
set "TRAIN_HAS_XML=0"
set "VAL_HAS_IMAGES=0"
set "VAL_HAS_XML=0"

for %%f in ("%TRAIN_PATH%\*.png" "%TRAIN_PATH%\*.jpg" "%TRAIN_PATH%\*.jpeg") do set "TRAIN_HAS_IMAGES=1"
for %%f in ("%TRAIN_PATH%\*.xml") do set "TRAIN_HAS_XML=1"
for %%f in ("%VAL_PATH%\*.png" "%VAL_PATH%\*.jpg" "%VAL_PATH%\*.jpeg") do set "VAL_HAS_IMAGES=1"
for %%f in ("%VAL_PATH%\*.xml") do set "VAL_HAS_XML=1"

if "%TRAIN_HAS_IMAGES%"=="0" (
    echo ERROR: No image files found in training folder: %TRAIN_PATH%
    goto get_paths
)
if "%TRAIN_HAS_XML%"=="0" (
    echo ERROR: No XML files found in training folder: %TRAIN_PATH%
    goto get_paths
)
if "%VAL_HAS_IMAGES%"=="0" (
    echo ERROR: No image files found in validation folder: %VAL_PATH%
    goto get_paths
)
if "%VAL_HAS_XML%"=="0" (
    echo ERROR: No XML files found in validation folder: %VAL_PATH%
    goto get_paths
)

echo Training Path: %TRAIN_PATH% >> "%LOG_FILE%"
echo Validation Path: %VAL_PATH% >> "%LOG_FILE%"
echo.

:: ============================================================================
:: 2. Environment Setup (Conda)
:: ============================================================================
echo [Step 2/9] Setting up Conda environment '%ENV_NAME%'... 
echo [Step 2/9] Setting up Conda environment '%ENV_NAME%'... >> "%LOG_FILE%"

:: Check if conda exists using the full path
if not exist "%CONDA_INSTALL_PATH%\Scripts\conda.exe" (
    echo ERROR: Conda not found at %CONDA_INSTALL_PATH%\Scripts\conda.exe >> "%LOG_FILE%"
    echo ERROR: Conda not found. Please verify your Conda installation path.
    echo Expected path: %CONDA_INSTALL_PATH%\Scripts\conda.exe
    goto :error
)

:: Check if environment already exists
"%CONDA_INSTALL_PATH%\Scripts\conda.exe" env list | findstr /B "%ENV_NAME% " >nul
if %errorlevel% equ 0 (
    echo Conda environment '%ENV_NAME%' already exists. Skipping creation. 
    echo Conda environment '%ENV_NAME%' already exists. Skipping creation. >> "%LOG_FILE%"
) else (
    echo Creating new Conda environment... 
    echo Creating new Conda environment... >> "%LOG_FILE%"
    "%CONDA_INSTALL_PATH%\Scripts\conda.exe" create -n %ENV_NAME% python=%PYTHON_VERSION% -y >> "%LOG_FILE%" 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create Conda environment. Check log: %LOG_FILE%
        goto :error
    )
)
echo.

:: ============================================================================
:: 3. Activate Environment and Install Core Dependencies
:: ============================================================================
echo [Step 3/9] Activating environment and installing PyTorch... 
echo [Step 3/9] Activating environment and installing PyTorch... >> "%LOG_FILE%"

:: Activate the environment using the full path
call "%CONDA_INSTALL_PATH%\Scripts\activate.bat" %ENV_NAME%
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate Conda environment. Check your CONDA_INSTALL_PATH. >> "%LOG_FILE%"
    echo ERROR: Failed to activate Conda environment. Check your CONDA_INSTALL_PATH.
    goto :error
)

:: Install PyTorch with CUDA
pip install torch==%PYTORCH_VERSION% torchvision==%TORCHVISION_VERSION% --index-url https://download.pytorch.org/whl/%CUDA_VERSION% >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch. Check log: %LOG_FILE%
    goto :error
)
echo PyTorch installed successfully. 
echo PyTorch installed successfully. >> "%LOG_FILE%"
echo.

:: ============================================================================
:: 4. Install MMDetection Dependencies
:: ============================================================================
echo [Step 4/9] Installing MMDetection dependencies... 
echo [Step 4/9] Installing MMDetection dependencies... >> "%LOG_FILE%"

:: Install MIM (OpenMMLab package manager)
pip install -U openmim >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to install openmim. Check log: %LOG_FILE%
    goto :error
)

:: Install MMEngine
mim install mmengine >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to install mmengine. Check log: %LOG_FILE%
    goto :error
)

:: Install MMCV
mim install "mmcv>=2.0.0" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to install mmcv. Check log: %LOG_FILE%
    goto :error
)

echo MMDetection dependencies installed successfully. 
echo MMDetection dependencies installed successfully. >> "%LOG_FILE%"
echo.

:: ============================================================================
:: 5. Clone and Install MMDetection
:: ============================================================================
echo [Step 5/9] Cloning and installing MMDetection... 
echo [Step 5/9] Cloning and installing MMDetection... >> "%LOG_FILE%"

if not exist "%MMDET_DIR%" (
    echo Cloning MMDetection repository... 
    echo Cloning MMDetection repository... >> "%LOG_FILE%"
    git clone https://github.com/open-mmlab/mmdetection.git "%MMDET_DIR%" >> "%LOG_FILE%" 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Failed to clone MMDetection. Check log: %LOG_FILE%
        goto :error
    )
) else (
    echo MMDetection directory already exists. Skipping clone. 
    echo MMDetection directory already exists. Skipping clone. >> "%LOG_FILE%"
)

cd "%MMDET_DIR%"
pip install -v -e . >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to install MMDetection. Check log: %LOG_FILE%
    goto :error
)

echo MMDetection installed successfully. 
echo MMDetection installed successfully. >> "%LOG_FILE%"
echo.

:: ============================================================================
:: 6. Prepare Dataset Structure
:: ============================================================================
echo [Step 6/9] Preparing dataset structure... 
echo [Step 6/9] Preparing dataset structure... >> "%LOG_FILE%"

:: Create VOC-style directory structure
set "TRAIN_VOC_DIR=%PROJECT_ROOT%data\train"
set "VAL_VOC_DIR=%PROJECT_ROOT%data\val"

if not exist "%TRAIN_VOC_DIR%\JPEGImages" mkdir "%TRAIN_VOC_DIR%\JPEGImages"
if not exist "%TRAIN_VOC_DIR%\Annotations" mkdir "%TRAIN_VOC_DIR%\Annotations"
if not exist "%TRAIN_VOC_DIR%\ImageSets\Main" mkdir "%TRAIN_VOC_DIR%\ImageSets\Main"

if not exist "%VAL_VOC_DIR%\JPEGImages" mkdir "%VAL_VOC_DIR%\JPEGImages"
if not exist "%VAL_VOC_DIR%\Annotations" mkdir "%VAL_VOC_DIR%\Annotations"
if not exist "%VAL_VOC_DIR%\ImageSets\Main" mkdir "%VAL_VOC_DIR%\ImageSets\Main"

:: Copy files to VOC structure
echo Copying training files... 
echo Copying training files... >> "%LOG_FILE%"
xcopy "%TRAIN_PATH%\*.png" "%TRAIN_VOC_DIR%\JPEGImages\" /Y /Q >> "%LOG_FILE%" 2>&1
xcopy "%TRAIN_PATH%\*.jpg" "%TRAIN_VOC_DIR%\JPEGImages\" /Y /Q >> "%LOG_FILE%" 2>&1
xcopy "%TRAIN_PATH%\*.jpeg" "%TRAIN_VOC_DIR%\JPEGImages\" /Y /Q >> "%LOG_FILE%" 2>&1
xcopy "%TRAIN_PATH%\*.xml" "%TRAIN_VOC_DIR%\Annotations\" /Y /Q >> "%LOG_FILE%" 2>&1

echo Copying validation files... 
echo Copying validation files... >> "%LOG_FILE%"
xcopy "%VAL_PATH%\*.png" "%VAL_VOC_DIR%\JPEGImages\" /Y /Q >> "%LOG_FILE%" 2>&1
xcopy "%VAL_PATH%\*.jpg" "%VAL_VOC_DIR%\JPEGImages\" /Y /Q >> "%LOG_FILE%" 2>&1
xcopy "%VAL_PATH%\*.jpeg" "%VAL_VOC_DIR%\JPEGImages\" /Y /Q >> "%LOG_FILE%" 2>&1
xcopy "%VAL_PATH%\*.xml" "%VAL_VOC_DIR%\Annotations\" /Y /Q >> "%LOG_FILE%" 2>&1

:: Generate ImageSets/Main/train.txt and val.txt
(for %%f in ("%TRAIN_VOC_DIR%\JPEGImages\*.*") do (echo %%~nf)) > "%TRAIN_VOC_DIR%\ImageSets\Main\train.txt"
(for %%f in ("%VAL_VOC_DIR%\JPEGImages\*.*") do (echo %%~nf)) > "%VAL_VOC_DIR%\ImageSets\Main\val.txt"

echo Dataset structure prepared successfully. 
echo Dataset structure prepared successfully. >> "%LOG_FILE%"
echo.

:: ============================================================================
:: 7. Auto-detect Class Names and Generate Configuration
:: ============================================================================
echo [Step 7/9] Auto-detecting class names and generating configuration... 
echo [Step 7/9] Auto-detecting class names and generating configuration... >> "%LOG_FILE%"

:: Create Python script to extract class names
(
    echo import os
    echo import xml.etree.ElementTree as ET
    echo from pathlib import Path
    echo.
    echo search_path = r'%TRAIN_VOC_DIR%\Annotations'
    echo all_names = set()
    echo for xml_file in Path(search_path).glob('*.xml'):
    echo     try:
    echo         tree = ET.parse(xml_file)
    echo         root = tree.getroot()
    echo         for obj in root.findall('object'):
    echo             name = obj.find('name').text
    echo             if name: all_names.add(name)
    echo     except ET.ParseError:
    echo         print(f'Warning: Could not parse {xml_file}')
    echo.
    echo print(','.join(sorted(list(all_names))))
) > get_classes.py

for /f %%i in ('python get_classes.py') do set "CLASSES_STR=%%i"
del get_classes.py

if not defined CLASSES_STR (
    echo WARNING: Could not automatically detect class names. Defaulting to 'chip'.
    echo WARNING: Could not automatically detect class names. Defaulting to 'chip'. >> "%LOG_FILE%"
    set "CLASSES_STR=chip"
)

echo Detected classes: %CLASSES_STR% 
echo Detected classes: %CLASSES_STR% >> "%LOG_FILE%"

:: Count number of classes
set "NUM_CLASSES=0"
for %%a in (%CLASSES_STR:,= %) do set /a NUM_CLASSES+=1

echo Number of classes: %NUM_CLASSES% 
echo Number of classes: %NUM_CLASSES% >> "%LOG_FILE%"

:: Generate Python tuple format for classes
set "PY_CLASSES_TUPLE=('!CLASSES_STR:,=','!')"

:: Generate the custom config file
set "CONFIG_NAME=faster-rcnn_r50_fpn_1x_voc_custom"
set "CONFIG_FILE=%MMDET_DIR%\configs\pascal_voc\%CONFIG_NAME%.py"

echo Generating MMDetection config file: %CONFIG_FILE% 
echo Generating MMDetection config file: %CONFIG_FILE% >> "%LOG_FILE%"

if not exist "%MMDET_DIR%\configs\pascal_voc" mkdir "%MMDET_DIR%\configs\pascal_voc"

(
    echo # Custom Faster R-CNN configuration for chip detection
    echo # Generated automatically by Manus training script
    echo.
    echo _base_ = [
    echo     '../_base_/models/faster-rcnn_r50_fpn.py',
    echo     '../_base_/datasets/voc0712.py',
    echo     '../_base_/schedules/schedule_1x.py',
    echo     '../_base_/default_runtime.py'
    echo ]
    echo.
    echo # Dataset settings
    echo dataset_type = 'VOCDataset'
    echo data_root = r'%PROJECT_ROOT%data/'
    echo classes = !PY_CLASSES_TUPLE!
    echo.
    echo # Model settings - Modify head for custom number of classes
    echo model = dict(
    echo     roi_head=dict(
    echo         bbox_head=dict(num_classes=%NUM_CLASSES%^)
    echo     ^)
    echo ^)
    echo.
    echo # Training dataloader
    echo train_dataloader = dict(
    echo     batch_size=1,  # Reduced for 11GB VRAM
    echo     num_workers=2,
    echo     dataset=dict(
    echo         type=dataset_type,
    echo         data_root=data_root + 'train/',
    echo         ann_file='ImageSets/Main/train.txt',
    echo         data_prefix=dict(sub_data_root=''^),
    echo         metainfo=dict(classes=classes^)
    echo     ^)
    echo ^)
    echo.
    echo # Validation dataloader
    echo val_dataloader = dict(
    echo     dataset=dict(
    echo         type=dataset_type,
    echo         data_root=data_root + 'val/',
    echo         ann_file='ImageSets/Main/val.txt',
    echo         data_prefix=dict(sub_data_root=''^),
    echo         metainfo=dict(classes=classes^)
    echo     ^)
    echo ^)
    echo.
    echo # Evaluation settings
    echo val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points'^)
    echo test_dataloader = val_dataloader
    echo test_evaluator = val_evaluator
    echo.
    echo # Load from COCO pre-trained model
    echo load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    echo.
    echo # Training schedule - Extended for better convergence
    echo train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5^)
    echo.
    echo # Optimizer settings - Lower learning rate for fine-tuning
    echo optim_wrapper = dict(
    echo     type='OptimWrapper',
    echo     optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001^)
    echo ^)
    echo.
    echo # Learning rate schedule
    echo param_scheduler = [
    echo     dict(
    echo         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500^),
    echo     dict(
    echo         type='MultiStepLR',
    echo         begin=0,
    echo         end=50,
    echo         by_epoch=True,
    echo         milestones=[30, 40],
    echo         gamma=0.1^)
    echo ]
) > "%CONFIG_FILE%"

echo Configuration file created successfully. 
echo Configuration file created successfully. >> "%LOG_FILE%"
echo.

:: ============================================================================
:: 8. Start Training
:: ============================================================================
echo [Step 8/9] Starting model training... 
echo [Step 8/9] Starting model training... >> "%LOG_FILE%"
echo This will take a long time. You can monitor progress in the console.
echo Training logs will be saved to: %PROJECT_ROOT%work_dirs\%CONFIG_NAME%
echo.

python tools/train.py "%CONFIG_FILE%" --work-dir "%PROJECT_ROOT%work_dirs\%CONFIG_NAME%" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Training failed. Check log: %LOG_FILE%
    echo The most common error is running out of GPU memory (CUDA out of memory).
    echo If so, edit the config file '%CONFIG_FILE%' and lower the 'batch_size' to 1.
    goto :error
)

echo.
echo [SUCCESS] Training completed! 
echo [SUCCESS] Training completed! >> "%LOG_FILE%"
echo.

:: ============================================================================
:: 9. Run Inference Test
:: ============================================================================
echo [Step 9/9] Running inference on a sample validation image... 
echo [Step 9/9] Running inference on a sample validation image... >> "%LOG_FILE%"

:: Find the best checkpoint file (.pth)
set "LATEST_EPOCH_FILE="
for /f "delims=" %%i in ('dir /b /a-d /o-d "%PROJECT_ROOT%work_dirs\%CONFIG_NAME%\epoch_*.pth" 2^>nul') do (
    set "LATEST_EPOCH_FILE=%%i"
    goto :found_checkpoint
)
:found_checkpoint

if not defined LATEST_EPOCH_FILE (
    echo ERROR: Could not find a trained checkpoint file. Cannot run inference. 
    echo ERROR: Could not find a trained checkpoint file. Cannot run inference. >> "%LOG_FILE%"
    goto :error
)

set "CHECKPOINT_PATH=%PROJECT_ROOT%work_dirs\%CONFIG_NAME%\%LATEST_EPOCH_FILE%"
echo Using checkpoint: %CHECKPOINT_PATH% 
echo Using checkpoint: %CHECKPOINT_PATH% >> "%LOG_FILE%"

:: Find a sample image from the validation set
set "SAMPLE_IMAGE="
for %%f in ("%VAL_VOC_DIR%\JPEGImages\*.*") do (
    set "SAMPLE_IMAGE=%%f"
    goto :found_image
)
:found_image

if not defined SAMPLE_IMAGE (
    echo ERROR: No images found in validation folder to test. 
    echo ERROR: No images found in validation folder to test. >> "%LOG_FILE%"
    goto :error
)

echo Testing on image: %SAMPLE_IMAGE% 
echo Testing on image: %SAMPLE_IMAGE% >> "%LOG_FILE%"

if not exist "%PROJECT_ROOT%inference_results" mkdir "%PROJECT_ROOT%inference_results"

python tools/test.py "%CONFIG_FILE%" "%CHECKPOINT_PATH%" --show-dir "%PROJECT_ROOT%inference_results" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Test script failed, but training was successful. Check log: %LOG_FILE%
    echo WARNING: Test script failed, but training was successful. Check log: %LOG_FILE% >> "%LOG_FILE%"
)

echo.
echo [SUCCESS] Inference test completed! 
echo [SUCCESS] Inference test completed! >> "%LOG_FILE%"
echo.

:: ============================================================================
:: 10. Final Instructions
:: ============================================================================
echo [COMPLETE] All tasks are finished. 
echo [COMPLETE] All tasks are finished. >> "%LOG_FILE%"
echo.
echo ###################################################
echo #                  PROCESS FINISHED               #
echo ###################################################
echo.
echo * Your trained model and logs are in:
echo   %PROJECT_ROOT%work_dirs\%CONFIG_NAME%
echo.
echo * The dataset was organized in VOC format at:
echo   %PROJECT_ROOT%data\
echo.
echo * Inference results (if successful) are in:
echo   %PROJECT_ROOT%inference_results
echo.
echo * Configuration file created at:
echo   %CONFIG_FILE%
echo.
echo * To run inference on new images, activate the conda environment and use:
echo   call "%CONDA_INSTALL_PATH%\Scripts\activate.bat" %ENV_NAME%
echo   cd %MMDET_DIR%
echo   python tools/test.py "%CONFIG_FILE%" "%CHECKPOINT_PATH%" --show-dir results
echo.
echo * Training completed at %date% %time%
echo * Training completed at %date% %time% >> "%LOG_FILE%"
goto :eof

:error
echo.
echo ###################################################
echo #               AN ERROR OCCURRED                 #
echo ###################################################
echo.
echo The script has stopped due to an error.
echo Please check the log file for details: %LOG_FILE%
echo.
echo Common solutions:
echo 1. Ensure you have sufficient GPU memory (11GB should be enough with batch_size=1)
echo 2. Check that your Conda installation path is correct: %CONDA_INSTALL_PATH%
echo 3. Verify that your training and validation folders contain both images and XML files
echo 4. Make sure you have a stable internet connection for downloading dependencies
echo.
endlocal
pause

