# YOLO Active Learning Annotation System

## Overview

This project provides an advanced system for object detection tasks, featuring an active learning loop with a sophisticated GUI annotation tool and a dedicated metrics visualization window. The system is designed to streamline the process of labeling image datasets, training YOLO models, and monitoring performance. It iteratively suggests annotations, allows for manual correction and labeling, retrains the model, and provides real-time feedback on training progress.

## Core Components

1.  **Active Learning Loop (`active_loop_V2.py`):**
    * Manages the overall active learning process.
    * Loads unlabeled images in batches.
    * Uses the current YOLO model to perform inference and generate pre-annotations for the annotation tool.
    * Launches the Annotation Tool V2 for user review and labeling of the pre-annotated batch.
    * Initiates model training in a separate process once new annotations are available.
    * Updates the primary inference model with the newly trained model upon successful completion.
    * Handles the lifecycle of the Metrics Window.
    * Clears previously annotated and saved images/labels from the working directories (`data/images/`, `data/labels/`) at the start of a new session. 

2.  **Annotation Tool V2 (`annotation_tool_V2.py`):**
    * A Tkinter-based GUI for detailed image annotation. 
    * Displays images with pre-loaded bounding boxes (if provided by the active learning loop). 
    * **Interactive Annotation:**
        * Draw new bounding boxes. 
        * Select, modify, and delete existing boxes. 
        * Assign labels to boxes from a predefined list (`labels.txt`). 
    * **Advanced Navigation & Viewing:**
        * Pan (Middle mouse button drag). 
        * Zoom (Mouse wheel, centered on cursor). 
        * Center and fit image to view. 
    * **Standalone Mode:** Can be run independently to annotate a directory of images. 
    * **Output:** Saves annotated images to `data/images/` and YOLO-format label files to `data/labels/`. When run standalone and completed, it can split the data into train/validation sets in the `dataset/` directory. 

## Features

* **Active Learning:** Efficiently label data by focusing on images where the model is least certain (implicitly, by correcting its predictions).
* **Parallel Training:** Model fine-tuning runs in a separate process, allowing annotation of the next batch to proceed without waiting. 
* **Interactive GUI Annotation (V2):**
    * User-friendly interface with mouse and keyboard controls. 
    * Zoom and pan capabilities for detailed annotation. 
    * Pre-loading of inference results to speed up labeling. 
    * Cycle through overlapping boxes with clicks. 
* **Real-time Metrics Visualization:** Monitor training progress and compare performance across runs. 
* **Standalone Annotation:** Flexibility to use the annotation tool independently of the active learning loop. 
* **Configurable:** Key parameters like batch size, image size, epochs, and paths are easily adjustable. 
* **Temporary Artifact Management:** Training runs, temporary datasets, and models are stored in `temp_training_artifacts/`. 
* **Graceful Exit & Cleanup:** Handles closing and attempts to clean up processes and temporary files. 

## Requirements

* Python 3.x
* Dependencies listed in `requirements.txt`. Install using:
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

```
object-detection/
├── active_loop_V2.py        # Main active-learning script (V2)
├── annotation_tool_V2.py    # GUI annotation tool (V2)
├── metrics_window.py        # Metrics display window script
├── data/
│   ├── images_unlabeled/    # Source directory for unlabeled input images
│   ├── images/              # Annotated images are copied here during active learning/standalone tool use
│   ├── labels/              # YOLO-format label files corresponding to images in data/images/
│   ├── completed/
│   │   ├── images/          # Original images that have been fully processed
│   │   ├── labels/          # Labels for non-tiled images
│   │   └── tiles/
│   │       ├── images/      # Completed tile images
│   │       └── labels/      # Completed tile labels
│   └── labels.txt           # Class index-to-name mapping
├── dataset/                 # Generated train/val splits (used by annotation_tool_V2.py standalone)
├── models/
│   └── yolov8s.pt           # Main YOLO model weights (updated by active_loop_V2.py)
├── temp_training_artifacts/ # Temporary files for active learning
│   ├── model_trained_run_X.pt # Models trained during specific runs
│   └── yolo_training_runs/  # Artifacts from each YOLO training session
│       └── run_X/           # Specific run artifacts
│           ├── results.csv      # Training metrics
│           ├── args.yaml        # Training arguments
│           └── training_data.yaml # YAML used for this training run
├── README.md                # This file
├── requirements.txt         # Python dependencies
```

## Configuration

Open `active_loop_V2.py` and adjust the following constants as needed: 

* `MODEL_PATH`: Path to the primary YOLO `.pt` model file.
* `UNLABELED_DIR`: Directory containing images to be labeled.
* `IMAGES_DIR`: Directory where images processed by the annotation tool are stored.
* `LABELS_DIR`: Directory where YOLO label files are stored.
* `LABELS_TXT_PATH`: Path to the file defining class names and their indices.
* `TEMP_MODEL_AND_RUN_DIR`: Root directory for storing temporary training artifacts.
* `BATCH_SIZE`: Number of images to process in each annotation batch.
* `IMG_SIZE`: Image size (e.g., 640) for YOLO model inference and training.
* `EPOCHS`: Number of epochs for each fine-tuning cycle in the active learning loop.
* `TRAIN_RATIO`: Ratio for splitting data into training and validation sets (e.g., 0.8 for 80% train).

For `annotation_tool_V2.py` (when run standalone): 

* It primarily uses paths relative to its location or pre-defined structure (e.g., `data/images_unlabeled`, `data/labels.txt`).
* `TRAIN_RATIO`: Used for splitting the dataset after standalone annotation.
* `MAX_CANVAS_VIEWPORT_WIDTH`/`HEIGHT`: Controls the initial maximum size of the image display area.

## Running the System

**1. Active Learning Loop with Annotation Tool V2 and Metrics Window:**

   Ensure your unlabeled images are in `data/images_unlabeled/` and `data/labels.txt` is correctly set up. The `models/yolov8s.pt` should exist (if not, the script attempts to load a default one).

   Navigate to the `object-detection` directory and run:
   ```bash
   python active_loop_V2.py
   ```

   **Workflow:**
   * The script will start, potentially clearing `data/images/` and `data/labels/`. 
   * The **Metrics Window** will appear. 
   * The **Annotation Tool V2** will launch for the first batch of images with pre-annotations. 
   * **Annotation Tool V2 Instructions:**
    * **Mouse:**
        * **Left Click & Drag:** Draw a new bounding box. 
        * **Left Click inside box:** Select a box. Repeated clicks on the same spot cycle through overlapping boxes. 
        * **Left Click & Drag on edges/corners of a selected box:** Resize the selected bounding box.
        * **Middle Mouse Drag:** Pan the image. 
        * **Mouse Wheel:** Zoom in/out, centered on the mouse cursor. 
    * **Keyboard Shortcuts:** * `n`: Save annotations for the current image and move to the next in the batch. If it's the last image, finishing the batch may trigger training. 
        * `r`: Reset (clear) all boxes on the current image. 
        * `d`: Delete the currently selected bounding box. 
        * `c`: Center the image and fit it to the current view (resets zoom/pan). 
        * `q`: Quit the annotation tool. If during an active learning session, this may abort the current session.
    * **Toggles:**
        * The tiling toggle will tile the image and only ouput the tiles images into the dataset
        * The Overlay toggle will load in the respecive csv file and display the analyst's drawings
   * After a batch is annotated and the Annotation Tool V2 is closed for that batch:
        * The main loop saves annotations. 
        * If new annotations were made, a new training process starts in the background using images from `data/images/` and labels from `data/labels/`. 
        * The **Metrics Window** will update with data from `temp_training_artifacts/yolo_training_runs/run_X/results.csv` as the training progresses and completes. 
        * You can continue annotating the next batch while training occurs. 
        * Once training is complete, the `models/yolov8s.pt` is updated.  Trained images/labels from `data/images/` and `data/labels/` are then removed to prepare for the next cycle with new data. 
   * The loop continues until all images in `data/images_unlabeled/` are processed or the user aborts. 

**2. Running Annotation Tool V2 Standalone:**

   Ensure your images are in `data/images_unlabeled/` and `data/labels.txt` is correctly formatted. 

   Navigate to the `object-detection` directory and run:
   ```bash
   python annotation_tool_V2.py
   ```
   * The tool will load images from `data/images_unlabeled/`. 
   * Use the GUI and keyboard shortcuts as described above. 
   * Upon completing annotations for an image (by pressing 'n'), the annotated image is saved to `data/images/` and the label file to `data/labels/`. 
   * After all images are processed or the tool is quit ('q'):
        * If annotations were made, the `split_dataset` function is called to organize the contents of `data/images/` and `data/labels/` into `dataset/images/{train,val}` and `dataset/labels/{train,val}`. 

## Outputs

* **`data/images/`**: Contains images that have been annotated (either through active learning or standalone tool). Cleared periodically by `active_loop_V2.py` after successful training cycles.
* **`data/labels/`**: Contains YOLO-format `.txt` label files corresponding to images in `data/images/`. Also cleared periodically.
* **`dataset/`**: When `annotation_tool_V2.py` is run standalone and completes, this directory is populated with `train/` and `val/` subdirectories for images and labels.
* **`models/yolov8s.pt`**: The main, continually updated YOLO model weights from the active learning loop.
* **`temp_training_artifacts/`**:
    * `model_trained_run_X.pt`: Snapshot of the model trained in a specific run of the active learning loop.
    * `yolo_training_runs/run_X/`: Contains detailed outputs for each training run (metrics, arguments, temporary dataset YAML).
        * `results.csv`: Key training and validation metrics per epoch.
        * `args.yaml`: Configuration arguments used for that specific training run.
        * `training_data.yaml`: The dataset configuration YAML file created for that training run.

## Important Notes

* **`labels.txt`:** This file is crucial. It maps class names to indices (e.g., `0:person`). Ensure it's present and correctly formatted in `data/labels.txt`. If an index is not specified (e.g., just `person`), the tool defaults to using the line number (0-indexed) as the class index. 
* **Initial Model:** If `models/yolov8s.pt` is not found when running `active_loop_V2.py`, the script will attempt to load a default `yolov8s.pt` from Ultralytics and save it to the `MODEL_PATH`. 
* **Temporary Datasets for Training:** The `active_loop_V2.py` creates temporary dataset structures and YAML files within `tempfile` directories for each training run, which are then referenced by the YOLO training process.  These temporary datasets are cleaned up after training. 
* **Metrics Persistence:** The Metrics Window is designed to stay open across multiple annotation and training cycles within a single `active_loop_V2.py` session. 
* **Previous Runs** are saved in `temp_training_artifacts/` and **will be overwritten** if ran again without saving them elsewhere 
* **Standalone Tool Dataset Splitting:** The `annotation_tool_V2.py`, when run standalone, will attempt to split the annotated data from `data/images/` and `data/labels/` into the `dataset/` directory only if annotations were actually made and saved.
