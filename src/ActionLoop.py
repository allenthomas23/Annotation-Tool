# Author: Allen Thomas

import shutil
from pathlib import Path
from ultralytics import YOLO
from AnnotationTool import AnnotationTool, split_dataset
import time
from multiprocessing import Process, Queue
import tempfile

from metrics_window import metrics_window_process
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models/yolov8s.pt"
UNLABELED_DIR = PROJECT_ROOT / "data/images_unlabeled"
IMAGES_DIR = PROJECT_ROOT / "data/images"
LABELS_DIR = PROJECT_ROOT / "data/labels"
DATASET_DIR = PROJECT_ROOT / "dataset"
COMPLETED_DIR = PROJECT_ROOT / "data/completed/images"
COMPLETED_LABELS_DIR = PROJECT_ROOT / "data/completed/labels"
COMPLETED_TILES_IMAGES_DIR = PROJECT_ROOT / "data/completed/tiles/images"
COMPLETED_TILES_LABELS_DIR = PROJECT_ROOT / "data/completed/tiles/labels"
LABELS_TXT_PATH = PROJECT_ROOT / "data/labels.txt"


TEMP_MODEL_AND_RUN_DIR = PROJECT_ROOT / "temp_training_artifacts"



BATCH_SIZE = 16
IMG_SIZE = 384
EPOCHS = 10
TRAIN_RATIO = 0.8



def ensure_dirs_exist():
    """Ensure all necessary directories exist, creating them if they don't."""
    UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_TILES_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_TILES_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_MODEL_AND_RUN_DIR.mkdir(parents=True, exist_ok=True)

    for sub in ['images', 'labels']:
        for split_type in ['train', 'val']:
            (DATASET_DIR / sub / split_type).mkdir(parents=True, exist_ok=True)


def run_training_process(
    current_all_images_dir_str: str,
    current_all_labels_dir_str: str,
    base_model_to_train_from_str: str,
    epochs_count: int,
    image_size_for_train: int,
    data_train_ratio: float,
    labels_txt_file_str: str,
    training_run_id: str,
    result_communication_queue: Queue
):
    """
    This function runs in a separate process to train the YOLO model.
    It prepares a temporary dataset, trains the model, and saves the trained model.
    """
    print(f"[Trainer-{training_run_id}] Training process started.")

    temp_dataset_root_path = Path(tempfile.mkdtemp(prefix=f"activelearn_train_data_{training_run_id}_"))

    trained_model_output_path = TEMP_MODEL_AND_RUN_DIR / f"model_trained_{training_run_id}.pt"

    src_images_path_obj = Path(current_all_images_dir_str)

    images_trained_in_this_run = [str(p.resolve()) for p in src_images_path_obj.iterdir() if p.is_file()]


    try:
        print(f"[Trainer-{training_run_id}] Preparing temporary dataset at: {temp_dataset_root_path}")
        split_dataset(
            src_images=src_images_path_obj,
            src_labels=Path(current_all_labels_dir_str),
            output_dir=temp_dataset_root_path,
            train_ratio=data_train_ratio
        )
        class_names_map = {}
        class_indices_list = []
        with open(labels_txt_file_str, 'r') as f_labels:
            for idx, line in enumerate(f_labels):
                line = line.strip()
                if not line: continue
                parts = line.split(':', 1)
                class_idx = int(parts[0]) if len(parts) > 1 and parts[0].isdigit() else idx
                class_name = parts[1].strip() if len(parts) > 1 else parts[0].strip()

                if class_idx not in class_names_map:
                    class_names_map[class_idx] = class_name
                    class_indices_list.append(class_idx)

        num_classes = len(class_indices_list)
        temp_yaml_file_path = temp_dataset_root_path / "training_data.yaml"
        with open(temp_yaml_file_path, 'w') as f_yaml:
            f_yaml.write(f"path: {str(temp_dataset_root_path.resolve())}\n")
            f_yaml.write(f"train: images/train\n")
            f_yaml.write(f"val: images/val\n")
            f_yaml.write(f"nc: {num_classes}\n")
            f_yaml.write("names:\n")
            for i in sorted(class_names_map.keys()):
                 f_yaml.write(f"  {i}: '{class_names_map[i]}'\n")

        print(f"[Trainer-{training_run_id}] Loading base model from: {base_model_to_train_from_str}")
        model = YOLO(base_model_to_train_from_str)

        print(f"[Trainer-{training_run_id}] Starting model training using {temp_yaml_file_path}...")
        yolo_runs_project_dir = TEMP_MODEL_AND_RUN_DIR / "yolo_training_runs"
        yolo_runs_project_dir.mkdir(parents=True, exist_ok=True)

        current_run_artifact_dir = yolo_runs_project_dir / training_run_id
        current_run_artifact_dir.mkdir(parents=True, exist_ok=True)

        if temp_yaml_file_path.exists():
            try:
                destination_yaml_path = current_run_artifact_dir / temp_yaml_file_path.name
                shutil.copy2(str(temp_yaml_file_path), str(destination_yaml_path))
                print(f"[Trainer-{training_run_id}] Copied {temp_yaml_file_path.name} to {destination_yaml_path}")
            except Exception as e_copy_yaml:
                print(f"[Trainer-{training_run_id}] Error copying {temp_yaml_file_path.name} to {current_run_artifact_dir}: {e_copy_yaml}")


        model.train(
            data=str(temp_yaml_file_path),
            epochs=epochs_count,
            imgsz=image_size_for_train,
            project=str(yolo_runs_project_dir),
            name=training_run_id,
            exist_ok=True,
            verbose=True
        )

        destination_yaml_target_dir = Path(yolo_runs_project_dir) / training_run_id
        if destination_yaml_target_dir.is_dir():
            destination_yaml_file_path = destination_yaml_target_dir / temp_yaml_file_path.name
            shutil.copy2(str(temp_yaml_file_path), str(destination_yaml_file_path))
            print(f"[Trainer-{training_run_id}] Copied training YAML to {destination_yaml_file_path}")
        else:
            print(f"[Trainer-{training_run_id}] WARNING: Could not copy training YAML. Destination directory {destination_yaml_target_dir} was not found.")

        model.save(str(trained_model_output_path))
        print(f"[Trainer-{training_run_id}] Training complete. Model saved to {trained_model_output_path}")
        result_communication_queue.put({
            'status': 'success',
            'model_path': str(trained_model_output_path),
            'id': training_run_id,
            'trained_image_paths': images_trained_in_this_run
        })
    except Exception as e:
        print(f"[Trainer-{training_run_id}] Error during training: {e}")
        import traceback
        traceback.print_exc()
        result_communication_queue.put({
            'status': 'error',
            'message': str(e),
            'id': training_run_id,
            'trained_image_paths': []
            })
    finally:
        if temp_dataset_root_path.exists():
            shutil.rmtree(temp_dataset_root_path)
            print(f"[Trainer-{training_run_id}] Cleaned up temporary dataset directory: {temp_dataset_root_path}")


def active_learning_loop():
    """
    Manages the active learning cycle.
    """
    ensure_dirs_exist()
    print("Starting active learning loop with parallel training")

    metric_window_queue = Queue()
    metric_window_process_obj = Process(
        target=metrics_window_process,
        args=(metric_window_queue,),
        daemon=True
    )
    metric_window_process_obj.start()
    print("Main Loop: Persistent window process started.")

    try:
        initial_model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"Error loading initial model from {MODEL_PATH}: {e}. Attempting to load default 'yolov8s.pt'.")
        if not MODEL_PATH.exists():
            try:
                initial_model = YOLO('yolov8s.pt')
                initial_model.save(str(MODEL_PATH))
                print(f"Default 'yolov8s.pt' loaded and saved to {MODEL_PATH}")
            except Exception as e_default:
                print(f"Failed to load/save default 'yolov8s.pt': {e_default}. Exiting.")
                if metric_window_process_obj.is_alive():
                    metric_window_queue.put("EXIT_PERSISTENT_WINDOW")
                    metric_window_process_obj.join(timeout=2)
                return
        else:
            print("Exiting due to model load failure.")
            if metric_window_process_obj.is_alive():
                metric_window_queue.put("EXIT_PERSISTENT_WINDOW")
                metric_window_process_obj.join(timeout=2)
            return

    all_unlabeled_image_paths = sorted([
        p.resolve() for p in UNLABELED_DIR.glob("*.*")
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ])

    if not all_unlabeled_image_paths:
        print("No unlabeled images found in 'data/images_unlabeled/'. Exiting.")
        if metric_window_process_obj.is_alive():
            metric_window_queue.put("EXIT_PERSISTENT_WINDOW")
            metric_window_process_obj.join(timeout=2)
        return
    print(f"Found {len(all_unlabeled_image_paths)} unlabeled images.")

    shared_state = {
        'current_inference_model': initial_model,
        'active_training_process': None,
        'current_active_training_id': None,
        'training_status_queue': Queue()
    }
    training_run_count = 0
    new_annotations_made_since_last_train_start = False

    def process_training_queue_callback():
        """Checks for and processes completed training runs."""
        q = shared_state['training_status_queue']
        if q.empty():
            return

        train_result = q.get()
        finished_run_id = train_result.get('id')
        print(f"Main Loop: Received training result for ID {finished_run_id}: {train_result.get('status')}")

        if train_result['status'] == 'success':
            newly_trained_model_path = Path(train_result['model_path'])
            if newly_trained_model_path.exists():
                if finished_run_id == shared_state['current_active_training_id']:
                    try:
                        shared_state['current_inference_model'] = YOLO(str(newly_trained_model_path))
                        shutil.copy2(str(newly_trained_model_path), MODEL_PATH)
                        print(f"Main Loop: Inference model updated from {newly_trained_model_path} for ID {finished_run_id}.")
                    except Exception as e_load:
                        print(f"Main Loop: Error loading new model from {newly_trained_model_path}: {e_load}")

            trained_images = train_result.get('trained_image_paths', [])
            if trained_images:
                original_files_moved = set()
                moved_originals, moved_labels, moved_tiles = 0, 0, 0
                for img_path_str in trained_images:
                    img_in_images_dir = Path(img_path_str)
                    is_tile = "_tile_" in img_in_images_dir.stem
                    original_stem = img_in_images_dir.stem.rsplit('_tile_', 1)[0] if is_tile else img_in_images_dir.stem

                    if original_stem not in original_files_moved:
                        for original_file in UNLABELED_DIR.glob(f'{original_stem}.*'):
                            if original_file.exists():
                                try:
                                    shutil.move(str(original_file), COMPLETED_DIR / original_file.name)
                                    moved_originals += 1
                                    original_files_moved.add(original_stem)
                                except Exception as e_move:
                                    print(f"Main Loop: Error moving {original_file.name}: {e_move}")
                                break
                    
                    label_in_labels_dir = LABELS_DIR / f"{img_in_images_dir.stem}.txt"
                    if label_in_labels_dir.exists():
                        try:
                            dest_dir = COMPLETED_TILES_LABELS_DIR if is_tile else COMPLETED_LABELS_DIR
                            shutil.move(str(label_in_labels_dir), dest_dir / label_in_labels_dir.name)
                            moved_labels += 1
                        except Exception as e_move_label:
                            print(f"Main Loop: Error moving label {label_in_labels_dir.name}: {e_move_label}")

                    try:
                        if img_in_images_dir.exists():
                            if is_tile:
                                shutil.move(str(img_in_images_dir), COMPLETED_TILES_IMAGES_DIR / img_in_images_dir.name)
                                moved_tiles += 1
                            else:
                                img_in_images_dir.unlink()
                    except OSError as e_move:
                        print(f"Main Loop: Error moving/deleting from data/images: {img_in_images_dir.name}: {e_move}")
                print(f"Main Loop: Moved {moved_originals} originals, {moved_tiles} tiles, and {moved_labels} labels to completed directories.")
        else:
            print(f"Main Loop: Training ID {finished_run_id} failed: {train_result.get('message')}")

        if finished_run_id == shared_state['current_active_training_id']:
            if shared_state['active_training_process']:
                shared_state['active_training_process'].join(timeout=0.1)
            shared_state['active_training_process'] = None
            shared_state['current_active_training_id'] = None
            print(f"Main Loop: Cleared active status for training ID {finished_run_id}.")

    try:
        while all_unlabeled_image_paths:
            active_proc = shared_state['active_training_process']
            if active_proc and not active_proc.is_alive():
                process_training_queue_callback()

            current_annotation_batch = all_unlabeled_image_paths[:BATCH_SIZE]
            all_unlabeled_image_paths = all_unlabeled_image_paths[BATCH_SIZE:]

            print(f"\nMain Loop: Preparing annotation for batch of {len(current_annotation_batch)} images.")
            
            try:
                inference_results = shared_state['current_inference_model'].predict(source=current_annotation_batch, imgsz=IMG_SIZE, verbose=False)
            except Exception as e_pred:
                print(f"Main Loop: Error during inference: {e_pred}. Skipping batch.")
                continue

            preload_annotations = {Path(r.path).resolve(): [(int(b.cls[0]), *b.xyxy[0].tolist()) for b in r.boxes] for r in inference_results}

            annotation_tool_instance = AnnotationTool(
                labels_dir=LABELS_DIR,
                labels_txt=LABELS_TXT_PATH,
                preload=preload_annotations,
                batch=current_annotation_batch,
                on_idle_callback=process_training_queue_callback
            )
            print("Main Loop: Launching Annotation Tool...")
            annotation_session_ok = annotation_tool_instance.run()

            if not annotation_session_ok:
                print("Main Loop: Annotation aborted by user. Shutting down...")
                if shared_state['active_training_process'] and shared_state['active_training_process'].is_alive():
                    shared_state['active_training_process'].terminate()
                    shared_state['active_training_process'].join()
                break

            print("Main Loop: Annotation for current batch completed.")
            new_annotations_made_since_last_train_start = True

            active_proc = shared_state['active_training_process']
            if not (active_proc and active_proc.is_alive()):
                if new_annotations_made_since_last_train_start and any(IMAGES_DIR.iterdir()):
                    training_run_count += 1
                    shared_state['current_active_training_id'] = f"run_{training_run_count}"
                    print(f"\nMain Loop: Starting new training process (ID: {shared_state['current_active_training_id']})...")
                    
                    p_args = (str(IMAGES_DIR.resolve()), str(LABELS_DIR.resolve()), str(MODEL_PATH.resolve()), EPOCHS, IMG_SIZE, TRAIN_RATIO, str(LABELS_TXT_PATH.resolve()), shared_state['current_active_training_id'], shared_state['training_status_queue'])
                    shared_state['active_training_process'] = Process(target=run_training_process, args=p_args)
                    shared_state['active_training_process'].start()
                    new_annotations_made_since_last_train_start = False
            else:
                print(f"Main Loop: Training ID {shared_state['current_active_training_id']} is ongoing.")

    finally:
        print("\nActive learning loop has concluded or was interrupted.")
        active_proc = shared_state['active_training_process']
        if active_proc and active_proc.is_alive():
            print("Waiting for final training process to complete...")
            active_proc.join(timeout=300)
            if active_proc.is_alive():
                print("Final training process timed out. Terminating.")
                active_proc.terminate()
                active_proc.join()
        
        process_training_queue_callback()

        if metric_window_process_obj.is_alive():
            print("Main Loop: Sending EXIT signal to persistent window.")
            metric_window_queue.put("EXIT_PERSISTENT_WINDOW")
            metric_window_process_obj.join(timeout=5)
            if metric_window_process_obj.is_alive():
                metric_window_process_obj.terminate()
        print("Main Loop: Persistent window process handled.")
        print("Exiting application.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    if not LABELS_TXT_PATH.exists():
        print(f"CRITICAL ERROR: '{LABELS_TXT_PATH.name}' not found. This file is essential.")
    else:
        print(f"Clearing contents of {IMAGES_DIR} and {LABELS_DIR} before starting a new session.")
        for d in [IMAGES_DIR, LABELS_DIR]:
            for item in d.glob('*'):
                try:
                    item.unlink() if item.is_file() else shutil.rmtree(item)
                except OSError as e:
                    print(f"Error deleting {item}: {e}")
        active_learning_loop()
