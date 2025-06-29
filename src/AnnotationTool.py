# Author: Allen Thomas
import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
import random
import shutil
from PIL import Image, ImageTk
import threading
import queue
import csv

Image.MAX_IMAGE_PIXELS = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_RATIO = 0.8

#maximum display dimensions for image data resizing
MAX_CANVAS_VIEWPORT_WIDTH = 1920
MAX_CANVAS_VIEWPORT_HEIGHT = 1080


def split_dataset(src_images, src_labels, output_dir, train_ratio=TRAIN_RATIO):
    """
    Split images and labels into train/val folders.
    Copies images from src_images and labels from src_labels
    into output_dir/images/{train,val} and output_dir/labels/{train,val}.
    Only copies label files if they exist.
    """
    src_images = Path(src_images)
    src_labels = Path(src_labels)
    base_out = Path(output_dir)

    #clear old dataset
    if base_out.exists():
        shutil.rmtree(base_out)

    for sub in ['images', 'labels']:
        for split in ['train', 'val']:
            (base_out / sub / split).mkdir(parents=True, exist_ok=True)

    #get all images
    img_paths = sorted([p for p in src_images.iterdir()
                        if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
    random.shuffle(img_paths)
    split_idx = int(len(img_paths) * train_ratio)
    train_imgs, val_imgs = img_paths[:split_idx], img_paths[split_idx:]

    #copy images and labels
    def copy_subset(img_list, img_dest, lbl_dest):
        for img_path_item in img_list:
            shutil.copy2(img_path_item, img_dest / img_path_item.name)
            lbl_file = src_labels / f"{img_path_item.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, lbl_dest / lbl_file.name)

    copy_subset(train_imgs, base_out / 'images' / 'train', base_out / 'labels' / 'train')
    copy_subset(val_imgs,   base_out / 'images' / 'val',   base_out / 'labels' / 'val')

    print(f"Dataset split complete: Train={len(train_imgs)}, Val={len(val_imgs)}")


class AnnotationTool:
    def __init__(
        self,
        images_dir: Path = None,
        labels_dir: Path = None,
        labels_txt: Path = None,
        preload: dict = None,
        batch: list = None,
        min_box_size: int = 5,
        on_idle_callback=None,
    ):
        self.root = tk.Tk()
        self.root.title("Annotation Tool")
        self._quitting_initiated = False

        self.on_idle_callback = on_idle_callback
        self.enable_tiling = True
        #threshold for minimum box size
        self.min_box_size = min_box_size
        #make sure multiple boxes can not be created
        self.awaiting_label = False
        #selecting box for deletion
        self.active_box = None

        self.original_pil_img = None
        #canvas image
        self.photo_image = None

        self.zoom_level = 1.0
        #original image view origin in original image coordinates
        self.view_orig_x0 = 0.0
        self.view_orig_y0 = 0.0

        self.is_panning = False
        self.last_pan_mouse_x = 0
        self.last_pan_mouse_y = 0

        self.is_zooming_active = False
        self.zoom_finalize_timer_id = None
        self.ZOOM_FINALIZE_DELAY = 250

        #image and canvas dimensions
        self.current_original_width = 0
        self.current_original_height = 0
        self.current_canvas_width = 1
        self.current_canvas_height = 1

        #queues for thread communication
        self.redraw_request_queue = queue.Queue(maxsize=1)
        self.redraw_result_queue = queue.Queue(maxsize=1)
        #signal for thread
        self.worker_thread_stop_event = threading.Event()
        self.image_processing_thread = None

        #canvas view rectangle for the original image
        self.displayed_orig_view_rect = None

        self.labels_txt = labels_txt if labels_txt and labels_txt.is_absolute() else PROJECT_ROOT / 'data' / 'labels.txt'
        self._load_labels()

        self.labels_dir = labels_dir if labels_dir and labels_dir.is_absolute() else PROJECT_ROOT / 'data' / 'labels'
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.images_out_dir = PROJECT_ROOT / 'data' / 'images'
        self.images_out_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_data_dir = PROJECT_ROOT / 'data' / 'overlays'
        self.overlay_data_dir.mkdir(parents=True, exist_ok=True)


        #preload is a dict with key as image path and value as label
        raw_preload = preload or {}
        self.preload = {}
        for k, v in raw_preload.items():
            keyp = Path(k).resolve()
            self.preload[keyp] = v

        #if there is a batch then it is not run standalone
        if batch:
            self.image_paths = batch
        elif images_dir:
            self.image_paths = sorted(
                [p.resolve() for p in Path(images_dir).iterdir()
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
            )
        else:
            default_images_dir = PROJECT_ROOT / 'data' / 'images_unlabeled'
            if default_images_dir.exists():
                self.image_paths = sorted(
                    [p.resolve() for p in default_images_dir.iterdir()
                     if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
                )
            else:
                messagebox.showerror("Error", "Image source not found.", parent=self.root)
                self.root.destroy()
                raise ValueError("Image source not found")

        self.index = 0
        #boxes is a list of tuples (label, x1, y1, x2, y2) that always is respective to original image coordinates
        self.boxes = []
        self.drawing = False
        #box coordinates
        self.start_pt_canvas = None
        self.current_pt_canvas = None
        self.run_aborted = False
        self.last_click_canvas_pos = None
        
        self.overlay_toggle_var = tk.BooleanVar(value=False)
        self.overlay_points = []
        
        self.is_resizing = False
        self.resizing_box_index = None
        self.resizing_edge = None
        self.resize_tolerance = 5
        
        self.hover_info = {'text': '', 'x': 0, 'y': 0, 'visible': False}
        self.active_overlay_point_index = -1

        self._setup_ui()
        self._setup_info_window()
        self._update_info_window()

        if not self.image_paths:
            messagebox.showinfo("Info", "No images found to annotate.", parent=self.root)
            self.root.after(100, self._quit_action_if_no_images)
            self.run_aborted = True

        self._start_image_processing_thread()
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.process_queue_after_id = self.root.after(100, self._process_redraw_queue) # Store ID for cancellation

    def _periodic_idle_check(self):
        """Periodically calls the provided callback to allow background processing."""
        if self.on_idle_callback:
            try:
                self.on_idle_callback()
            except Exception as e:
                print(f"Error during idle callback: {e}")
        
        if hasattr(self, 'root') and self.root and self.root.winfo_exists():
            self.root.after(2000, self._periodic_idle_check) # Check every 2 seconds

    def _show_initial_instructions_and_settings(self):
        """Shows a dialog with instructions and a setting to enable tiling."""
        self.proceed_with_annotation = False

        dialog = tk.Toplevel(self.root)
        dialog.title("Instructions & Settings")
        dialog.transient(self.root) 
        dialog.grab_set()

        instruction_text = (
            "Left Click & Drag: Draw box\n"
            "Left Click inside box: Select/Cycle selection\n"
            "Middle Mouse Drag: Pan image\n"
            "Mouse Wheel: Zoom image (centered on cursor)\n\n"
            "Keys:\n"
            "n: Next image (Save current)\n"
            "r: Reset boxes on current image\n"
            "d: Delete selected box\n"
            "c: Center and fit image (reset zoom/pan)\n"
            "q: Quit annotation tool"
        )
        tk.Label(dialog, text=instruction_text, justify=tk.LEFT, padx=10, pady=10).pack()

        tiling_var = tk.BooleanVar(value=self.enable_tiling)
        chk_tiling = tk.Checkbutton(dialog, text="Enable Image Tiling (384x384 tiles, 1/3 x overlap, 50% y overlap)",
                                    variable=tiling_var)
        chk_tiling.pack(pady=5)

        def on_ok():
            self.enable_tiling = tiling_var.get()
            self.proceed_with_annotation = True # User confirmed to start
            dialog.destroy()

        ok_button = tk.Button(dialog, text="Start Annotating", command=on_ok, width=20)
        ok_button.pack(pady=10)
        dialog.bind("<Return>", lambda event: on_ok()) # Allow Enter key to confirm

        def on_dialog_close_button(): # Handle [X] button closure
            # If closed via [X], treat it as not proceeding unless "Start" was somehow clicked
            if not self.proceed_with_annotation:
                self.run_aborted = True # This signals the run() method to abort
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close_button)

        # Calculate position to center dialog over the main window (if visible) or screen
        self.root.update_idletasks()
        dialog.update_idletasks()

        dialog_width = dialog.winfo_reqwidth()
        dialog_height = dialog.winfo_reqheight()

        if self.root.winfo_viewable():
            root_x = self.root.winfo_x()
            root_y = self.root.winfo_y()
            root_width = self.root.winfo_width()
            root_height = self.root.winfo_height()
            position_x = root_x + (root_width // 2) - (dialog_width // 2)
            position_y = root_y + (root_height // 2) - (dialog_height // 2)
        else: # Fallback to screen center if root window isn't ready
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            position_x = (screen_width // 2) - (dialog_width // 2)
            position_y = (screen_height // 2) - (dialog_height // 2)

        dialog.geometry(f"{dialog_width}x{dialog_height}+{position_x}+{position_y}")

        dialog.focus_set() # Set focus to the dialog
        self.root.wait_window(dialog) # Wait for the dialog to close

        return self.proceed_with_annotation

    def _on_canvas_resize(self, event):
        """Handles canvas resize events to update view
        Use: When the user manually resized the canvas or the code changes the dimensions"""
        #if the canvas size is different from the current canvas size then it updates the size
        if self.current_canvas_width != event.width or self.current_canvas_height != event.height:
            self.current_canvas_width = event.width
            self.current_canvas_height = event.height
            if self.original_pil_img:
                self._center_and_fit_image_action()

    def _quit_action_if_no_images(self):
        """Quits the application if no images are available for annotation."""
        if not self.image_paths and not self.run_aborted:
             self._quit_action()

    def _start_image_processing_thread(self):
        """Starts the image processing worker thread that handles cropping and resizing."""
        if self.image_processing_thread is None or not self.image_processing_thread.is_alive():
            self.worker_thread_stop_event.clear()
            self.image_processing_thread = threading.Thread(target=self._image_processing_worker, daemon=True)
            self.image_processing_thread.start()

    def _image_processing_worker(self):
        """Worker thread: crops and resizes image portion for display and returns it through the queue."""
        while not self.worker_thread_stop_event.is_set():
            try:

                request = self.redraw_request_queue.get(timeout=0.1)
                if request is None: break

                #unpack request parameters
                original_pil_img = request['original_pil_img']
                #vertical and horizontal offset from the original image top left corner
                view_orig_x0 = request['view_orig_x0']
                view_orig_y0 = request['view_orig_y0']
                current_zoom = request['current_zoom']
                target_canvas_w = request['target_canvas_w']
                target_canvas_h = request['target_canvas_h']
                is_panning = request['is_panning']
                is_zooming_active = request['is_zooming_active']
                orig_img_w = request['orig_img_w']
                orig_img_h = request['orig_img_h']


                if original_pil_img is None or target_canvas_w <=0 or target_canvas_h <=0:
                    self.redraw_result_queue.put({'pil_image_display': None, 'params': request, 'rendered_for_orig_rect': None})
                    self.redraw_request_queue.task_done()
                    continue

                #dimensions of the section to be cropped in original image coordinates
                orig_crop_width = target_canvas_w / current_zoom if current_zoom > 0 else orig_img_w
                orig_crop_height = target_canvas_h / current_zoom if current_zoom > 0 else orig_img_h

                #cropped coordinates
                crop_x0 = view_orig_x0
                crop_y0 = view_orig_y0
                crop_x1 = view_orig_x0 + orig_crop_width
                crop_y1 = view_orig_y0 + orig_crop_height

                #clamp to stay in boundaries
                crop_x0 = max(0, crop_x0)
                crop_y0 = max(0, crop_y0)
                crop_x1 = min(orig_img_w, crop_x1)
                crop_y1 = min(orig_img_h, crop_y1)

                rendered_for_orig_rect = (crop_x0, crop_y0, crop_x1, crop_y1)

                #grey placeholder if crop rectangle is invalid
                if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                    display_pil = Image.new("RGB", (target_canvas_w, target_canvas_h), "gray")
                else:
                    #crop original image to the rectangle
                    cropped_pil = original_pil_img.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))
                    #use lower quality resampling filter if panning or zooming is active then load in high quality
                    resampling_filter = Image.Resampling.BILINEAR if is_panning or is_zooming_active else Image.Resampling.LANCZOS
                    display_pil = cropped_pil.resize((target_canvas_w, target_canvas_h), resampling_filter)

                self.redraw_result_queue.put({'pil_image_display': display_pil, 'params': request, 'rendered_for_orig_rect': rendered_for_orig_rect})
            #nothing in queue, continue to next iteration
            except queue.Empty:
                continue
            except Exception as e_outer:
                print(f"Worker: Outer loop error: {e_outer}")
            finally:
                if 'request' in locals() and self.redraw_request_queue.unfinished_tasks > 0:
                    try: self.redraw_request_queue.task_done()
                    except ValueError: pass


    def _process_redraw_queue(self):
        """Processes the redraw result queue received from worker to update the canvas with the latest image.
        then schedules itself to run again after a short delay."""
        try:
            if not self.root.winfo_exists(): return
            #might need a try block here
            result = self.redraw_result_queue.get_nowait()

            pil_image_display = result.get('pil_image_display')
            self.displayed_orig_view_rect = result.get('rendered_for_orig_rect')


            if pil_image_display:
                try:
                    self.photo_image = ImageTk.PhotoImage(image=pil_image_display, master=self.canvas)
                    self._perform_canvas_update()
                except Exception as e_photo:
                    print(f"MainThread: Error creating PhotoImage: {e_photo}")
                    self.photo_image = None
                    self._perform_canvas_update()
            elif 'error' in result:
                 print(f"MainThread: Received error from worker: {result['error']}")
                 self.photo_image = None
                 self._perform_canvas_update()
            else:
                 self.photo_image = None
                 self._perform_canvas_update()

            self.redraw_result_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in _process_redraw_queue: {e}")
        #schedule next proccesing
        if self.root.winfo_exists():
            self.process_queue_after_id = self.root.after(30, self._process_redraw_queue)

    def _request_redraw(self):
        """Requests a redraw of the canvas by putting parameters into the queue for the image processing worker."""
        if not self.original_pil_img or not hasattr(self, 'root') or not self.root.winfo_exists():
            return

        # Ensure canvas dimensions are current before requesting a redraw based on them
        # This is important if canvas size changed and <Configure> event hasn't updated self.current_canvas_width/height yet
        if self.canvas.winfo_exists():
            c_width = self.canvas.winfo_width()
            c_height = self.canvas.winfo_height()
            if c_width > 1: self.current_canvas_width = c_width
            if c_height > 1: self.current_canvas_height = c_height

        if self.current_canvas_width <= 1 or self.current_canvas_height <= 1 : return


        try:
            #should only have one request in the queue at a time
            while not self.redraw_request_queue.empty():
                self.redraw_request_queue.get_nowait()
                self.redraw_request_queue.task_done()
        except queue.Empty:
            pass

        current_request_params = {
            'original_pil_img': self.original_pil_img,
            'view_orig_x0': self.view_orig_x0,
            'view_orig_y0': self.view_orig_y0,
            'current_zoom': self.zoom_level,
            'target_canvas_w': self.current_canvas_width,
            'target_canvas_h': self.current_canvas_height,
            'is_panning': self.is_panning,
            'is_zooming_active': self.is_zooming_active,
            'orig_img_w': self.current_original_width,
            'orig_img_h': self.current_original_height
        }
        try:
            self.redraw_request_queue.put_nowait(current_request_params)
        except queue.Full:
            print("Warning: Redraw request queue was full. Request might be lost.")


    def _perform_canvas_update(self):
        """Updates the canvas with the current image and boxes."""
        if not self.canvas.winfo_exists(): return

        self.canvas.delete("all")

        if self.photo_image:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image, tags="image_on_canvas")

        #should always have values
        if not self.displayed_orig_view_rect:
            self._update_info_window()
            return
            
        if self.overlay_toggle_var.get() and self.overlay_points:
            for i, (x_orig, y_orig, name) in enumerate(self.overlay_points):
                cx, cy = self._original_to_canvas_coords_cropped_view(x_orig, y_orig)
                radius = 3 if i == self.active_overlay_point_index else 1
                color = "red" if i == self.active_overlay_point_index else "yellow"
                self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=color, outline=color, tags="overlay_point")


        #draw each box in the canvas
        for i, (lbl, x1_o, y1_o, x2_o, y2_o) in enumerate(self.boxes):
            color = "red" if i == self.active_box else "green"
            line_w = 3 if i == self.active_box else 2

            # Convert original coordinates to canvas coordinates so the stay fixed with zoom/pan
            cx1, cy1 = self._original_to_canvas_coords_cropped_view(x1_o, y1_o)
            cx2, cy2 = self._original_to_canvas_coords_cropped_view(x2_o, y2_o)

            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=line_w, tags="box")
            #label in the top left corner of the box
            fl_text_x = min(cx1, cx2)
            fl_text_y = min(cy1, cy2) - 5

            self.canvas.create_text(int(fl_text_x), int(fl_text_y), text=lbl, fill="white", anchor=tk.SW, font=("Arial", 10, "bold"))

        #show the box dragging as drawn
        if self.drawing and self.start_pt_canvas and self.current_pt_canvas:
            self.canvas.create_rectangle(
                self.start_pt_canvas[0], self.start_pt_canvas[1],
                self.current_pt_canvas[0], self.current_pt_canvas[1],
                outline="blue", width=1, dash=(4, 2), tags="drawing_box"
            )
        
        # Draw the hover label for overlay points
        if self.hover_info['visible']:
            x, y = self.hover_info['x'], self.hover_info['y']
            text = self.hover_info['text']
            # A rough estimate for text width
            text_width = len(text) * 7 
            self.canvas.create_rectangle(x, y - 15, x + text_width, y + 5, fill="white", outline="black", tags="hover_label")
            self.canvas.create_text(x + 5, y - 5, text=text, anchor=tk.W, tags="hover_label", fill="black")

        self._update_info_window()


    def _original_to_canvas_coords_cropped_view(self, original_x, original_y):
        """Converts original image coordinates to canvas coordinates based on the currently displayed cropped view."""
        if not self.displayed_orig_view_rect or self.current_canvas_width <= 0 or self.current_canvas_height <= 0:
            return 0, 0

        crop_x0, crop_y0, crop_x1, crop_y1 = self.displayed_orig_view_rect

        #distance between corners of crop
        orig_view_w = crop_x1 - crop_x0
        orig_view_h = crop_y1 - crop_y0

        if orig_view_w <= 0 or orig_view_h <= 0: return 0,0

        #These values give you the coordinates of the (original_x, original_y) point
        # as if the top-left corner of the crop (crop_x0, crop_y0) was the origin (0,0)
        relative_x_in_crop = original_x - crop_x0
        relative_y_in_crop = original_y - crop_y0

        canvas_x = (relative_x_in_crop / orig_view_w) * self.current_canvas_width
        canvas_y = (relative_y_in_crop / orig_view_h) * self.current_canvas_height

        return canvas_x, canvas_y

    def _canvas_to_original_coords(self, canvas_x, canvas_y):
        """Converts canvas coordinates to original image coordinates based on the currently displayed cropped view."""
        # Fallback if displayed_orig_view_rect isn't set
        if not self.displayed_orig_view_rect or self.current_canvas_width <= 0 or self.current_canvas_height <= 0:
            if self.zoom_level == 0: return self.view_orig_x0, self.view_orig_y0
            orig_x = self.view_orig_x0 + (canvas_x / self.zoom_level)
            orig_y = self.view_orig_y0 + (canvas_y / self.zoom_level)
            return orig_x, orig_y

        #Essentially the reverse of _original_to_canvas_coords_cropped_view
        crop_x0, crop_y0, crop_x1, crop_y1 = self.displayed_orig_view_rect
        orig_view_w = crop_x1 - crop_x0
        orig_view_h = crop_y1 - crop_y0

        if orig_view_w <= 0 or orig_view_h <= 0: return crop_x0, crop_y0

        prop_x = canvas_x / self.current_canvas_width
        prop_y = canvas_y / self.current_canvas_height

        original_x = crop_x0 + (prop_x * orig_view_w)
        original_y = crop_y0 + (prop_y * orig_view_h)

        return original_x, original_y

    def _original_to_canvas_coords_for_info(self, original_x, original_y):
        """Converts original image coordinates to canvas coordinates for displaying in the info window."""
        return self._original_to_canvas_coords_cropped_view(original_x, original_y)


    def _setup_ui(self):
        """Sets up the main UI components of the annotation tool."""
        #canvas allows for drawing boxes
        self.canvas = tk.Canvas(self.root, background="black", takefocus=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        #mouse callbacks
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)

        self.canvas.bind("<ButtonPress-2>", self._on_pan_press)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)

        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        
        self.canvas.bind("<Motion>", self._on_canvas_hover)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        #container
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, pady=5)

        tk.Button(control_frame, text="Next (n)", command=self._next_image_action).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Reset (r)", command=self._reset_action).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Delete (d)", command=self._delete_action).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Center/Fit (c)", command=self._center_and_fit_image_action).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(control_frame, text="Show Overlay", variable=self.overlay_toggle_var, command=self._on_overlay_toggle).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Quit (q)", command=self._handle_user_abort).pack(side=tk.LEFT, padx=5)
        
        self.status_bar = tk.Label(self.root, text="Image: N/A", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        #hotkeys
        self.root.bind('n', lambda event: self._next_image_action())
        self.root.bind('r', lambda event: self._reset_action())
        self.root.bind('d', lambda event: self._delete_action())
        self.root.bind('c', lambda event: self._center_and_fit_image_action())
        self.root.bind('q', lambda event: self._handle_user_abort())
        self.root.bind('<Left>', lambda event: self._on_arrow_pan_horizontal(-1))
        self.root.bind('<Right>', lambda event: self._on_arrow_pan_horizontal(1))
        self.root.bind('<Up>', lambda event: self._on_arrow_pan_vertical(-1))
        self.root.bind('<Down>', lambda event: self._on_arrow_pan_vertical(1))

        self.root.protocol("WM_DELETE_WINDOW", self._handle_user_abort)
        #a fix for now for hotkeys to work -- look into this later
        self.canvas.focus_set()

    def _on_overlay_toggle(self):
        self._request_redraw()

    def _load_overlay_data(self, img_path):
        self.overlay_points = []
        csv_path = self.overlay_data_dir / f"{img_path.stem}.csv"
        print(f"Loading overlay data from {csv_path}")
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            x = float(row['x'])
                            y = float(row['y'])
                            # Get name, default to empty string if not present
                            name = row.get('names', '') 
                            self.overlay_points.append((x, y, name))
                        except (ValueError, IndexError, KeyError) as e:
                            print(f"Warning: Could not parse row in {csv_path}: {row}. Error: {e}")
            except Exception as e:
                print(f"Error loading overlay CSV {csv_path}: {e}")

        print(f"Loaded {len(self.overlay_points)} overlay points from {csv_path}")


    def _on_canvas_hover(self, event):
        """Handles mouse hover events on the canvas to show overlay point names."""
        if not self.overlay_toggle_var.get() or not self.overlay_points:
            return

        found_point = False
        min_dist_sq = (5*5)

        new_active_index = -1

        for i, (x_orig, y_orig, name) in enumerate(self.overlay_points):
            if not name:
                continue

            cx, cy = self._original_to_canvas_coords_cropped_view(x_orig, y_orig)
            dist_sq = (event.x - cx)**2 + (event.y - cy)**2

            if dist_sq < min_dist_sq:
                self.hover_info['text'] = name
                self.hover_info['x'] = event.x + 10 # Offset from cursor
                self.hover_info['y'] = event.y - 10 # Offset from cursor
                self.hover_info['visible'] = True
                new_active_index = i
                found_point = True
                break

        if not found_point and self.hover_info['visible']:
            self.hover_info['visible'] = False
            new_active_index = -1

        if self.active_overlay_point_index != new_active_index:
            self.active_overlay_point_index = new_active_index
            self._request_redraw()


    def _on_canvas_leave(self, event):
        if self.hover_info['visible']:
            self.hover_info['visible'] = False
            self.active_overlay_point_index = -1
            self._request_redraw()

    def _setup_info_window(self):
        """Sets up the information window that displays details about the current image and selected boxes."""
        self.info_window = tk.Toplevel(self.root)
        self.info_window.title("Annotation Information")
        self.info_window.geometry("350x250")
        self.info_window.resizable(False, False)

        self.info_window.protocol("WM_DELETE_WINDOW", self.info_window.withdraw)

        info_frame = tk.Frame(self.info_window, padx=10, pady=10)
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.info_image_text = tk.StringVar(value="Image: N/A")
        tk.Label(info_frame, textvariable=self.info_image_text, anchor="w").pack(fill="x", pady=2)

        self.info_zoom_pan_text = tk.StringVar(value="Zoom: 1.00x, Pan Orig_Coord: (0,0)")
        tk.Label(info_frame, textvariable=self.info_zoom_pan_text, anchor="w").pack(fill="x", pady=2)

        # Separator
        tk.Frame(info_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=(5, 5))
        self.info_box_index_text = tk.StringVar(value="Selected Box: N/A")
        tk.Label(info_frame, textvariable=self.info_box_index_text, anchor="w").pack(fill="x", pady=2)
        self.info_box_label_text = tk.StringVar(value="Label: N/A")
        tk.Label(info_frame, textvariable=self.info_box_label_text, anchor="w").pack(fill="x", pady=2)
        self.info_box_coords_orig_text = tk.StringVar(value="Orig Coords: N/A")
        tk.Label(info_frame, textvariable=self.info_box_coords_orig_text, anchor="w").pack(fill="x", pady=2)
        self.info_box_size_orig_text = tk.StringVar(value="Orig Size: N/A")
        tk.Label(info_frame, textvariable=self.info_box_size_orig_text, anchor="w").pack(fill="x", pady=2)

        # Separator
        tk.Frame(info_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=(5, 5))
        self.info_box_coords_disp_text = tk.StringVar(value="Disp Coords (Canvas): N/A")
        tk.Label(info_frame, textvariable=self.info_box_coords_disp_text, anchor="w").pack(fill="x", pady=2)
        self.info_box_size_disp_text = tk.StringVar(value="Disp Size (Canvas): N/A")
        tk.Label(info_frame, textvariable=self.info_box_size_disp_text, anchor="w").pack(fill="x", pady=2)

    def _update_info_window(self):
        """Updates the information window with details about the current image and selected boxes."""
        if not hasattr(self, 'info_window'): return
        try:
            if not self.info_window.winfo_exists():
                 self._setup_info_window()
            elif not self.info_window.winfo_viewable():
                self.info_window.deiconify()
        except tk.TclError:
            return

        if self.image_paths and self.index < len(self.image_paths):
            img_p = self.image_paths[self.index]
            self.info_image_text.set(f"Image: {img_p.name} ({self.index + 1}/{len(self.image_paths)})")
        else:
            self.info_image_text.set("Image: N/A")

        self.info_zoom_pan_text.set(f"Zoom: {self.zoom_level:.2f}x, View Orig TL: ({int(self.view_orig_x0)},{int(self.view_orig_y0)})")

        #output all info
        if self.active_box is not None and self.boxes and self.active_box < len(self.boxes) and self.current_original_width > 0:
            lbl, x1_o, y1_o, x2_o, y2_o = self.boxes[self.active_box]
            self.info_box_index_text.set(f"Selected Box: #{self.active_box}")
            self.info_box_label_text.set(f"Label: {lbl}")

            norm_x1_o, norm_x2_o = min(x1_o, x2_o), max(x1_o, x2_o)
            norm_y1_o, norm_y2_o = min(y1_o, y2_o), max(y1_o, y2_o)
            self.info_box_coords_orig_text.set(f"Orig Coords: ({int(norm_x1_o)},{int(norm_y1_o)})-({int(norm_x2_o)},{int(norm_y2_o)})")
            w_o, h_o = int(norm_x2_o - norm_x1_o), int(norm_y2_o - norm_y1_o)
            self.info_box_size_orig_text.set(f"Orig Size: W={w_o}, H={h_o}")

            cx1, cy1 = self._original_to_canvas_coords_for_info(norm_x1_o, norm_y1_o)
            cx2, cy2 = self._original_to_canvas_coords_for_info(norm_x2_o, norm_y2_o)
            self.info_box_coords_disp_text.set(f"Disp Coords: ({int(cx1)},{int(cy1)})-({int(cx2)},{int(cy2)})")
            w_d, h_d = int(abs(cx2 - cx1)), int(abs(cy2 - cy1))
            self.info_box_size_disp_text.set(f"Disp Size: W={w_d}, H={h_d}")
        else:
            self.info_box_index_text.set("Selected Box: N/A")
            self.info_box_label_text.set("Label: N/A")
            self.info_box_coords_orig_text.set("Orig Coords: N/A")
            self.info_box_size_orig_text.set("Orig Size: N/A")
            self.info_box_coords_disp_text.set("Disp Coords: N/A")
            self.info_box_size_disp_text.set("Disp Size: N/A")

    def _load_labels(self):
        """Loads labels from the labels.txt file into a dictionary."""
        #dict - name : idx (original index from file)
        self.label_to_idx = {}
        parsed_items = []
        if self.labels_txt.exists():
            with open(self.labels_txt, 'r') as f:
                for line_num_for_default_idx, line_content in enumerate(f):
                    stripped_line = line_content.strip()
                    if not stripped_line: continue
                    name_to_parse, index_to_parse = "", -1
                    if ':' in stripped_line:
                        num_str, lbl_name_part = stripped_line.split(':', 1)
                        name_to_parse = lbl_name_part.strip()
                        try: index_to_parse = int(num_str)
                        except ValueError:
                            print(f"Warning: Invalid index '{num_str}' for label '{name_to_parse}'. Using default {line_num_for_default_idx}.")
                            index_to_parse = line_num_for_default_idx
                    else:
                        #default to line number
                        name_to_parse, index_to_parse = stripped_line, line_num_for_default_idx
                    if name_to_parse in self.label_to_idx:
                        print(f"Warning: Duplicate label '{name_to_parse}'. Using first index {self.label_to_idx[name_to_parse]}.")
                    else:
                        self.label_to_idx[name_to_parse] = index_to_parse
                        parsed_items.append((index_to_parse, name_to_parse))
            parsed_items.sort()
            self.valid_labels = [item[1] for item in parsed_items]
        else:
            print(f"labels.txt not found at {self.labels_txt}")
            self.valid_labels = []

    def _on_mouse_press(self, event):
        """Handles mouse press events, initiating drawing or resizing mode."""
        self.canvas.focus_set()
        if self.awaiting_label or not self.original_pil_img or self.is_panning:
            return

        cx, cy = event.x, event.y
        
        # Check for resize grab first the corners have priority
        for i, (lbl, x1_o, y1_o, x2_o, y2_o) in enumerate(self.boxes):
            cx1, cy1 = self._original_to_canvas_coords_cropped_view(x1_o, y1_o)
            cx2, cy2 = self._original_to_canvas_coords_cropped_view(x2_o, y2_o)

            # Ensure coordinates are ordered for calculations
            norm_cx1, norm_cx2 = min(cx1, cx2), max(cx1, cx2)
            norm_cy1, norm_cy2 = min(cy1, cy2), max(cy1, cy2)

            # Check corners
            on_tl = (abs(cx - norm_cx1) < self.resize_tolerance and abs(cy - norm_cy1) < self.resize_tolerance)
            on_tr = (abs(cx - norm_cx2) < self.resize_tolerance and abs(cy - norm_cy1) < self.resize_tolerance)
            on_bl = (abs(cx - norm_cx1) < self.resize_tolerance and abs(cy - norm_cy2) < self.resize_tolerance)
            on_br = (abs(cx - norm_cx2) < self.resize_tolerance and abs(cy - norm_cy2) < self.resize_tolerance)
            
            # Check edges
            on_left = abs(cx - norm_cx1) < self.resize_tolerance and norm_cy1 <= cy <= norm_cy2
            on_right = abs(cx - norm_cx2) < self.resize_tolerance and norm_cy1 <= cy <= norm_cy2
            on_top = abs(cy - norm_cy1) < self.resize_tolerance and norm_cx1 <= cx <= norm_cx2
            on_bottom = abs(cy - norm_cy2) < self.resize_tolerance and norm_cx1 <= cx <= norm_cx2

            cursor = ""
            if on_tl: self.resizing_edge = 'top_left'; cursor = "size_nw_se"
            elif on_br: self.resizing_edge = 'bottom_right'; cursor = "size_nw_se"
            elif on_tr: self.resizing_edge = 'top_right'; cursor = "size_ne_sw"
            elif on_bl: self.resizing_edge = 'bottom_left'; cursor = "size_ne_sw"
            elif on_left: self.resizing_edge = 'left'; cursor = "sb_h_double_arrow"
            elif on_right: self.resizing_edge = 'right'; cursor = "sb_h_double_arrow"
            elif on_top: self.resizing_edge = 'top'; cursor = "sb_v_double_arrow"
            elif on_bottom: self.resizing_edge = 'bottom'; cursor = "sb_v_double_arrow"
            else:
                continue

            self.is_resizing = True
            self.active_box = i
            self.resizing_box_index = i
            self.canvas.config(cursor=cursor)
            return

        # If not resizing, proceed with drawing
        self.drawing = True
        self.start_pt_canvas = (cx, cy)
        self.current_pt_canvas = (cx, cy)
        self.potential_click_start_pos_canvas = (cx, cy)
        self._request_redraw()

    def _on_mouse_drag(self, event):
        """Handles mouse drag events to update the current box being drawn or resized."""
        if self.is_resizing and self.resizing_box_index is not None:
            lbl, x1_o, y1_o, x2_o, y2_o = self.boxes[self.resizing_box_index]
            orig_x, orig_y = self._canvas_to_original_coords(event.x, event.y)

            if self.resizing_edge == 'left':
                self.boxes[self.resizing_box_index] = (lbl, orig_x, y1_o, x2_o, y2_o)
            elif self.resizing_edge == 'right':
                self.boxes[self.resizing_box_index] = (lbl, x1_o, y1_o, orig_x, y2_o)
            elif self.resizing_edge == 'top':
                self.boxes[self.resizing_box_index] = (lbl, x1_o, orig_y, x2_o, y2_o)
            elif self.resizing_edge == 'bottom':
                self.boxes[self.resizing_box_index] = (lbl, x1_o, y1_o, x2_o, orig_y)
            elif self.resizing_edge == 'top_left':
                self.boxes[self.resizing_box_index] = (lbl, orig_x, orig_y, x2_o, y2_o)
            elif self.resizing_edge == 'top_right':
                self.boxes[self.resizing_box_index] = (lbl, x1_o, orig_y, orig_x, y2_o)
            elif self.resizing_edge == 'bottom_left':
                self.boxes[self.resizing_box_index] = (lbl, orig_x, y1_o, x2_o, orig_y)
            elif self.resizing_edge == 'bottom_right':
                self.boxes[self.resizing_box_index] = (lbl, x1_o, y1_o, orig_x, orig_y)

            self._request_redraw()
            return
            
        if self.drawing and self.start_pt_canvas and not self.awaiting_label and not self.is_panning:
            self.current_pt_canvas = (event.x, event.y)
            self.canvas.config(cursor="cross")
            self._request_redraw()

    def _on_mouse_release(self, event):
        """Handles mouse release. Finalizes new box or resize operation."""
        if self.is_resizing:
            self.is_resizing = False
            self.resizing_box_index = None
            self.resizing_edge = None
            self.canvas.config(cursor="")
            self._request_redraw()
            return

        if not self.drawing or not self.start_pt_canvas or self.awaiting_label or self.is_panning:
            if self.drawing:
                 self.drawing = False
                 self.start_pt_canvas = None
                 self.current_pt_canvas = None
            self.canvas.config(cursor="")
            return

        self.drawing = False
        _x1_canvas_start, _y1_canvas_start = self.start_pt_canvas
        _x2_canvas_end, _y2_canvas_end = event.x, event.y

        drag_distance_x = abs(_x2_canvas_end - _x1_canvas_start)
        drag_distance_y = abs(_y2_canvas_end - _y1_canvas_start)

        if drag_distance_x > self.min_box_size or drag_distance_y > self.min_box_size:
            self.awaiting_label = True
            lbl = self._prompt_label()
            if lbl:
                drawn_box_cx1 = min(_x1_canvas_start, _x2_canvas_end)
                drawn_box_cy1 = min(_y1_canvas_start, _y2_canvas_end)
                drawn_box_cx2 = max(_x1_canvas_start, _x2_canvas_end)
                drawn_box_cy2 = max(_y1_canvas_start, _y2_canvas_end)

                orig_x1, orig_y1 = self._canvas_to_original_coords(drawn_box_cx1, drawn_box_cy1)
                orig_x2, orig_y2 = self._canvas_to_original_coords(drawn_box_cx2, drawn_box_cy2)
                self.boxes.append((lbl, orig_x1, orig_y1, orig_x2, orig_y2))
                self.active_box = len(self.boxes) - 1
            else:
                 pass
            self.awaiting_label = False
            self.last_click_canvas_pos = None
        else:
            cx_click, cy_click = self.potential_click_start_pos_canvas
            orig_click_x, orig_click_y = self._canvas_to_original_coords(cx_click, cy_click)

            clicked_indices = []
            for i, (lbl, x1_o, y1_o, x2_o, y2_o) in enumerate(self.boxes):
                box_orig_x_min, box_orig_x_max = min(x1_o, x2_o), max(x1_o, x2_o)
                box_orig_y_min, box_orig_y_max = min(y1_o, y2_o), max(y1_o, y2_o)
                if box_orig_x_min <= orig_click_x <= box_orig_x_max and \
                   box_orig_y_min <= orig_click_y <= box_orig_y_max:
                    clicked_indices.append(i)

            if clicked_indices:
                if (hasattr(self, 'last_click_canvas_pos') and
                        self.last_click_canvas_pos == (cx_click, cy_click) and
                        self.active_box is not None and self.active_box in clicked_indices):
                    current_selection_idx_in_list = clicked_indices.index(self.active_box)
                    next_selection_idx_in_list = (current_selection_idx_in_list + 1) % len(clicked_indices)
                    self.active_box = clicked_indices[next_selection_idx_in_list]
                else:
                    self.active_box = clicked_indices[-1]
                self.last_click_canvas_pos = (cx_click, cy_click)
            else:
                self.active_box = None
                self.last_click_canvas_pos = None

        self.start_pt_canvas = None
        self.current_pt_canvas = None
        self.canvas.config(cursor="")
        self._request_redraw()
        self._update_info_window()

    def _on_pan_press(self, event):
        """Handles mouse press events for panning the image."""
        self.canvas.focus_set()
        self.is_panning = True
        self.last_pan_mouse_x = event.x
        self.last_pan_mouse_y = event.y
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, event):
        """Handles mouse drag events for panning the image."""
        if not self.is_panning: return

        dx_canvas = event.x - self.last_pan_mouse_x
        dy_canvas = event.y - self.last_pan_mouse_y

        # Pan amount in original image pixels (scaled by current zoom level)
        if self.zoom_level > 0:
            self.view_orig_x0 -= dx_canvas / self.zoom_level
            self.view_orig_y0 -= dy_canvas / self.zoom_level

        # Clamp pan
        if self.zoom_level > 0 and self.current_original_width > 0 and self.current_original_height > 0:
            orig_visible_w = self.current_canvas_width / self.zoom_level
            orig_visible_h = self.current_canvas_height / self.zoom_level

            self.view_orig_x0 = max(0, min(self.view_orig_x0, self.current_original_width - orig_visible_w))
            self.view_orig_y0 = max(0, min(self.view_orig_y0, self.current_original_height - orig_visible_h))

            if self.current_original_width <= orig_visible_w :
                self.view_orig_x0 = (self.current_original_width - orig_visible_w) / 2.0
            if self.current_original_height <= orig_visible_h :
                self.view_orig_y0 = (self.current_original_height - orig_visible_h) / 2.0

        self.last_pan_mouse_x = event.x
        self.last_pan_mouse_y = event.y
        self._request_redraw()
        self._update_info_window()

    def _on_pan_release(self, event):
        """Handles mouse release events for panning the image."""
        if self.is_panning:
            self.is_panning = False
            self.canvas.config(cursor="")
            self._request_redraw()
        else:
            self.is_panning = False
            self.canvas.config(cursor="")
    def _on_arrow_pan_horizontal(self, direction_multiplier):
        """Handles horizontal panning via arrow keys."""
        if not self.original_pil_img or self.is_panning or self.is_zooming_active:
            return

        self.canvas.focus_set()

        PAN_AMOUNT_ORIGINAL = 50

        delta_x_original = direction_multiplier * PAN_AMOUNT_ORIGINAL

        self.view_orig_x0 += delta_x_original

        if self.zoom_level > 0 and self.current_original_width > 0:

            original_visible_width = self.current_canvas_width / self.zoom_level
            max_x0 = self.current_original_width - original_visible_width

            if original_visible_width >= self.current_original_width:
                self.view_orig_x0 = (self.current_original_width - original_visible_width) / 2.0
            else:
                self.view_orig_x0 = max(0, min(self.view_orig_x0, max_x0))
        else:
            self.view_orig_x0 = 0

        self._request_redraw()
        self._update_info_window()

    def _on_arrow_pan_vertical(self, direction_multiplier):
            """Handles vertical panning via arrow keys."""
            if not self.original_pil_img or self.is_panning or self.is_zooming_active:
                return

            self.canvas.focus_set()
            PAN_AMOUNT_ORIGINAL_Y = 50
            delta_y_original = direction_multiplier * PAN_AMOUNT_ORIGINAL_Y
            self.view_orig_y0 += delta_y_original
            if self.zoom_level > 0 and self.current_original_height > 0:
                original_visible_height = self.current_canvas_height / self.zoom_level
                max_y0 = self.current_original_height - original_visible_height
                if original_visible_height >= self.current_original_height:
                    self.view_orig_y0 = (self.current_original_height - original_visible_height) / 2.0
                else:
                    self.view_orig_y0 = max(0, min(self.view_orig_y0, max_y0))
            else:
                self.view_orig_y0 = 0

            self._request_redraw()
            self._update_info_window()


    def _finalize_zoom_render(self):
        """Finalizes the zoom rendering after a delay to avoid flickering."""
        self.is_zooming_active = False
        self.zoom_finalize_timer_id = None
        if hasattr(self, 'original_pil_img') and self.original_pil_img and self.root.winfo_exists():
            self._request_redraw()


    def _on_mouse_wheel(self, event):
        """Handles mouse wheel events for zooming in and out."""
        self.canvas.focus_set()
        #canvas mouse coordinates
        canvas_mx, canvas_my = event.x, event.y

        #cancel any pending zoom finalize timer
        if self.zoom_finalize_timer_id:
            self.root.after_cancel(self.zoom_finalize_timer_id)
            self.zoom_finalize_timer_id = None

        self.is_zooming_active = True

        orig_mouse_x_before, orig_mouse_y_before = self._canvas_to_original_coords(canvas_mx, canvas_my)

        old_zoom_level = self.zoom_level
        if event.num == 5 or event.delta < 0:
            self.zoom_level /= 1.1
        elif event.num == 4 or event.delta > 0:
            self.zoom_level *= 1.1

        min_zoom = 0.05
        max_zoom = 20.0
        self.zoom_level = max(min_zoom, min(self.zoom_level, max_zoom))
        if self.zoom_level == 0 : self.zoom_level = old_zoom_level

        #keep the same relative mouse pos
        self.view_orig_x0 = orig_mouse_x_before - (canvas_mx / self.zoom_level)
        self.view_orig_y0 = orig_mouse_y_before - (canvas_my / self.zoom_level)

        #Clamp pan after zoom
        if self.zoom_level > 0 :
            orig_visible_w = self.current_canvas_width / self.zoom_level
            orig_visible_h = self.current_canvas_height / self.zoom_level

            self.view_orig_x0 = max(0, min(self.view_orig_x0, self.current_original_width - orig_visible_w))
            self.view_orig_y0 = max(0, min(self.view_orig_y0, self.current_original_height - orig_visible_h))
            if self.current_original_width <= orig_visible_w : self.view_orig_x0 = (self.current_original_width - orig_visible_w) / 2.0
            if self.current_original_height <= orig_visible_h : self.view_orig_y0 = (self.current_original_height - orig_visible_h) / 2.0

        self._request_redraw()
        self._update_info_window()

        self.zoom_finalize_timer_id = self.root.after(self.ZOOM_FINALIZE_DELAY, self._finalize_zoom_render)


    def _prompt_label(self):
        """Prompts the user to select a label from the available labels."""
        dialog = tk.Toplevel(self.root)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.title("Select Label")

        dialog.update_idletasks()
        parent_x, parent_y = self.root.winfo_x(), self.root.winfo_y()
        parent_w, parent_h = self.root.winfo_width(), self.root.winfo_height()
        dw, dh = 250, 120
        x_c = parent_x + (parent_w // 2) - (dw // 2)
        y_c = parent_y + (parent_h // 2) - (dh // 2)
        dialog.geometry(f'{dw}x{dh}+{x_c}+{y_c}')

        var = tk.StringVar()
        combo_values = self.valid_labels if isinstance(self.valid_labels, list) else []
        combo = ttk.Combobox(dialog, textvariable=var, values=combo_values, state="readonly")
        combo.pack(padx=10, pady=10)
        if combo_values: combo.current(0)
        combo.focus()

        res = {"label": ""}
        def apply():
            sel = var.get()
            if not sel and combo_values:
                messagebox.showwarning("No Label", "Please select a label or cancel.", parent=dialog)
                return
            res["label"] = sel
            dialog.destroy()
        def cancel():
            res["label"] = ""
            dialog.destroy()

        btn_f = tk.Frame(dialog)
        btn_f.pack(pady=5)
        tk.Button(btn_f, text="Apply", width=8, command=apply).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_f, text="Cancel", width=8, command=cancel).pack(side=tk.LEFT, padx=5)
        dialog.bind("<Return>", lambda e: apply())
        dialog.bind("<Escape>", lambda e: cancel())

        self.root.wait_window(dialog)
        return res["label"]

    def _save_annotations(self, img_path, w_orig, h_orig):
        """Saves the annotations for the current image to a .txt file in YOLO format."""
        out_file = self.labels_dir / f"{img_path.stem}.txt"
        if not self.boxes:
            if out_file.exists(): out_file.unlink()
            return

        with open(out_file, 'w') as f:
            for lbl_name, x1, y1, x2, y2 in self.boxes:
                class_idx = self.label_to_idx.get(lbl_name)
                if class_idx is None:
                    print(f"Warning: Label '{lbl_name}' not found in mapping for {img_path.name}. Skipping.")
                    continue
                if w_orig == 0 or h_orig == 0:
                    print(f"Warning: Original dimensions for {img_path.name} are zero. Skipping.")
                    continue

                norm_x1, norm_x2 = min(x1, x2), max(x1, x2)
                norm_y1, norm_y2 = min(y1, y2), max(y1, y2)
                cx = ((norm_x1 + norm_x2) / 2) / w_orig
                cy = ((norm_y1 + norm_y2) / 2) / h_orig
                bw = (norm_x2 - norm_x1) / w_orig
                bh = (norm_y2 - norm_y1) / h_orig
                f.write(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"Saved annotations to {out_file}")

    def _center_and_fit_image_action(self):
        """Resets zoom and centers the image to fit the canvas view."""
        if not self.original_pil_img or not self.canvas.winfo_exists(): return

        self.is_panning = False
        self.is_zooming_active = False
        if self.zoom_finalize_timer_id:
            self.root.after_cancel(self.zoom_finalize_timer_id)
            self.zoom_finalize_timer_id = None

        canvas_w = self.current_canvas_width
        canvas_h = self.current_canvas_height

        if self.current_original_width <= 0 or self.current_original_height <=0 or canvas_w <=0 or canvas_h <=0:
            self.zoom_level = 1.0
        else:
            zoom_to_fit_x = canvas_w / self.current_original_width
            zoom_to_fit_y = canvas_h / self.current_original_height
            self.zoom_level = min(zoom_to_fit_x, zoom_to_fit_y)


        min_zoom = 0.05
        max_zoom = 20.0
        self.zoom_level = max(min_zoom, min(self.zoom_level, max_zoom))
        if self.zoom_level == 0 : self.zoom_level = min_zoom

        #Calculate view_orig_x0, y0 to center the content
        #Content width/height in original pixels at this new zoom_level
        content_width_orig_at_new_zoom = canvas_w / self.zoom_level if self.zoom_level > 0 else self.current_original_width
        content_height_orig_at_new_zoom = canvas_h / self.zoom_level if self.zoom_level > 0 else self.current_original_height

        self.view_orig_x0 = (self.current_original_width - content_width_orig_at_new_zoom) / 2.0
        self.view_orig_y0 = (self.current_original_height - content_height_orig_at_new_zoom) / 2.0

        #Clamp view origin if image is smaller than viewport at this zoom so it's centered or at 0,0
        if self.current_original_width <= content_width_orig_at_new_zoom:
             self.view_orig_x0 = (self.current_original_width - content_width_orig_at_new_zoom) / 2.0
        else: #Image is wider than viewport, ensure we don't pan beyond edges
            self.view_orig_x0 = max(0, min(self.view_orig_x0, self.current_original_width - content_width_orig_at_new_zoom))

        if self.current_original_height <= content_height_orig_at_new_zoom:
            self.view_orig_y0 = (self.current_original_height - content_height_orig_at_new_zoom) / 2.0
        else: #Image is taller than viewport
            self.view_orig_y0 = max(0, min(self.view_orig_y0, self.current_original_height - content_height_orig_at_new_zoom))

        self._request_redraw()


    def _load_current_image(self):
        """Loads the current image and prepares it for display."""
        if self.index >= len(self.image_paths):
            self._finish_annotation()
            return

        img_p = self.image_paths[self.index]
        print(f"Attempting to load image from: {img_p}")
        self._load_overlay_data(img_p)
        
        try:
            original_cv_img = cv2.imread(str(img_p))
            if original_cv_img is None: raise IOError(f"Image not loaded or path incorrect: {img_p}")
            self.original_pil_img = Image.fromarray(cv2.cvtColor(original_cv_img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {img_p.name}\n{e}", parent=self.root)
            self.index += 1
            self.root.after(10, self._load_current_image)
            return

        self.current_original_width, self.current_original_height = self.original_pil_img.size
        if self.current_original_width == 0 or self.current_original_height == 0:
            messagebox.showerror("Error", f"Image {img_p.name} has zero dimensions. Skipping.", parent=self.root)
            self.index += 1
            self.root.after(10, self._load_current_image)
            return

        img_ratio = self.current_original_width / self.current_original_height if self.current_original_height != 0 else 1

        target_canvas_w = MAX_CANVAS_VIEWPORT_WIDTH
        target_canvas_h = MAX_CANVAS_VIEWPORT_HEIGHT

        #if the image is smaller just output it as is
        if self.current_original_width < target_canvas_w and self.current_original_height < target_canvas_h:
            target_canvas_w = self.current_original_width
            target_canvas_h = self.current_original_height
        else:
            # Fit the image to the canvas while maintaining aspect ratio
            vp_ratio = target_canvas_w / target_canvas_h if target_canvas_h !=0 else 1
            if img_ratio > vp_ratio:
                target_canvas_w = MAX_CANVAS_VIEWPORT_WIDTH
                target_canvas_h = int(target_canvas_w / img_ratio) if img_ratio != 0 else MAX_CANVAS_VIEWPORT_HEIGHT
            else:
                target_canvas_h = MAX_CANVAS_VIEWPORT_HEIGHT
                target_canvas_w = int(target_canvas_h * img_ratio)

        self.current_canvas_width = max(100, int(target_canvas_w))
        self.current_canvas_height = max(100, int(target_canvas_h))

        self.canvas.config(width=self.current_canvas_width, height=self.current_canvas_height)

        adj_win_w = self.current_canvas_width + 40
        adj_win_h = self.current_canvas_height + 100
        self.root.geometry(f"{adj_win_w}x{adj_win_h}")
        #force tkinter to update the window size
        self.root.update_idletasks()


        #These self.current_canvas_width/height will be passed to worker thread.
        actual_w = self.canvas.winfo_width()
        actual_h = self.canvas.winfo_height()
        if actual_w > 1 : self.current_canvas_width = actual_w
        if actual_h > 1 : self.current_canvas_height = actual_h

        #add the boxes from inference if available
        self.boxes = []; self.active_box = None
        if img_p.resolve() in self.preload:
            for cls_idx_preload, x1_o, y1_o, x2_o, y2_o in self.preload[img_p.resolve()]:
                lbl_name = ""
                for name, original_idx in self.label_to_idx.items():
                    if original_idx == cls_idx_preload: lbl_name = name; break
                if lbl_name: self.boxes.append((lbl_name, int(x1_o), int(y1_o), int(x2_o), int(y2_o)))

        self._center_and_fit_image_action()

        self.status_bar.config(text=f"Image: {img_p.name} ({self.index + 1}/{len(self.image_paths)})")
        #needed curerrently for hotkeys to work
        self.canvas.focus_force()


    def _next_image_action(self):
        # Ensure we are operating on a loaded image
        if not (self.index < len(self.image_paths) and self.original_pil_img and self.current_original_width > 0):
            self.index += 1 # Increment to avoid getting stuck if image was bad
            if self.index < len(self.image_paths):
                self._load_current_image()
            else:
                self._finish_annotation()
            return

        current_img_path = self.image_paths[self.index]

        if self.enable_tiling and self.boxes:
            processed_tiles_with_boxes = self._process_and_save_tiles(
                original_pil_img=self.original_pil_img.copy(),
                original_boxes=list(self.boxes),
                original_img_path=current_img_path,
                tile_size=(384, 384),
                overlap_percent_x=1/3,
                overlap_percent_y=0.5
            )

            if processed_tiles_with_boxes:

                # Ensure any existing label file for the original image is removed.
                original_label_file = self.labels_dir / f"{current_img_path.stem}.txt"
                if original_label_file.exists():
                    try:
                        original_label_file.unlink()
                    except OSError as e:
                        print(f"Error removing original label file {original_label_file} after successful tiling: {e}")
            else:
                # Tiling enabled. annotations present. but no valid tiles with boxes were made.
                error_message = (
                    f"Image {current_img_path.name} had annotations and tiling was enabled, but no valid "
                    f"tiles with fully contained bounding boxes were generated. This image and its "
                    f"annotations will not be saved to the output directories."
                )
                print(f"ERROR: {error_message}")
                if hasattr(self, 'root') and self.root.winfo_exists():
                    messagebox.showerror("Tiling Output Issue", error_message, parent=self.root)

                #Ensure original label file is removed
                original_label_file = self.labels_dir / f"{current_img_path.stem}.txt"
                if original_label_file.exists():
                    try:
                        original_label_file.unlink()
                    except OSError as e:
                        print(f"Error removing original label file {original_label_file} due to tiling output issue: {e}")

        else:

            self._save_annotations(current_img_path, self.current_original_width, self.current_original_height)

            if self.boxes:
                try:
                    self.original_pil_img.save(str(self.images_out_dir / current_img_path.name))
                except Exception as e:
                    print(f"Error saving original image {current_img_path.name}: {e}")

        # Move to the next image
        self.index += 1
        if self.index < len(self.image_paths):
            self._load_current_image()
        else:
            self._finish_annotation()

    def _reset_action(self):
        """Resets the boxes for the current image."""
        self.boxes = []
        self.active_box = None
        self._request_redraw()

    def _delete_action(self):
        """Deletes the currently active box."""
        if self.active_box is not None and 0 <= self.active_box < len(self.boxes):
            self.boxes.pop(self.active_box)
            self.active_box = None
            self._request_redraw()
        elif self.active_box is not None:
             self.active_box = None
             self._request_redraw()

    def _handle_user_abort(self):
        """Sets the abort flag and then calls the main quit action."""
        self.run_aborted = True
        self._quit_action()

    def _quit_action(self):
        """Handles the quit action, cleaning up resources and stopping threads."""
        if hasattr(self, '_quitting_initiated') and self._quitting_initiated:
            return
        self._quitting_initiated = True # Set the flag

        self.worker_thread_stop_event.set()
        try:

            self.redraw_request_queue.put_nowait(None)
        except queue.Full:
            pass

        # Destroy info_window if it exists and is valid
        if hasattr(self, 'info_window') and self.info_window:
            try:
                if self.info_window.winfo_exists():
                    self.info_window.destroy()
            except tk.TclError:
                pass
        self.info_window = None

        self.original_pil_img = None
        self.photo_image = None

        # Join worker thread
        if self.image_processing_thread and self.image_processing_thread.is_alive():
            self.image_processing_thread.join(timeout=0.5) # Wait for thread to finish
            if self.image_processing_thread.is_alive():
                print("Warning: Image processing thread did not join quickly.")
        self.image_processing_thread = None # Clear the reference

        # Destroy root window if it exists and is valid
        if hasattr(self, 'root') and self.root: # Check if attribute exists and is not None
            try:
                if self.root.winfo_exists(): # Check if Tk widget actually exists
                    # Cancel any pending 
                    if hasattr(self, 'zoom_finalize_timer_id') and self.zoom_finalize_timer_id:
                        self.root.after_cancel(self.zoom_finalize_timer_id)
                    self.zoom_finalize_timer_id = None

                    if hasattr(self, 'process_queue_after_id') and self.process_queue_after_id:
                        self.root.after_cancel(self.process_queue_after_id)
                    self.process_queue_after_id = None

                    self.root.quit() 
                    self.root.destroy()
            except tk.TclError:
                pass
        self.root = None 


    def _finish_annotation(self):
        """Finalizes the annotation process, saving any remaining annotations and cleaning up."""
        if hasattr(self, 'root') and self.root.winfo_exists():
            messagebox.showinfo("Annotation Complete", "All images processed.", parent=self.root)
        else:
            print("Annotation Complete. All images processed.")
        self._quit_action()

    def run(self):
        """Runs the annotation tool."""
        if not self.image_paths:
            print("No images to process.")
            self.worker_thread_stop_event.set()
            try: self.redraw_request_queue.put_nowait(None)
            except queue.Full: pass
            if self.image_processing_thread and self.image_processing_thread.is_alive():
                self.image_processing_thread.join(timeout=0.1)
            if hasattr(self, 'root') and self.root.winfo_exists(): self.root.destroy()
            return False

        if not self._show_initial_instructions_and_settings():
            self._quit_action() # Ensure full cleanup if dialog is dismissed
            return False

        self.root.after(10, self._load_current_image)
        self.root.after(2000, self._periodic_idle_check)

        self.root.mainloop()
        if not self.run_aborted :
            self._quit_action()
        return not self.run_aborted
    def _process_and_save_tiles(self, original_pil_img: Image.Image, original_boxes: list, original_img_path: Path, tile_size: tuple, overlap_percent_x: float,
                                overlap_percent_y: float,max_padding_ratio_allowed: float = 0) -> bool:
        tiles_generated_with_boxes = False
        tile_w, tile_h = tile_size

        if tile_w <= 0 or tile_h <= 0:
            print(f"Warning: tile_w ({tile_w}) or tile_h ({tile_h}) is zero or negative for {original_img_path.name}. Skipping tiling.")
            return False

        overlap_pixels_w = int(tile_w * overlap_percent_x)
        overlap_pixels_h = int(tile_h * overlap_percent_y)

        step_w = tile_w - overlap_pixels_w
        step_h = tile_h - overlap_pixels_h

        if step_w <= 0:
            step_w = tile_w
        if step_h <= 0:
            step_h = tile_h

        if step_w <= 0 or step_h <= 0 :
            print(f"Error: Cannot tile {original_img_path.name} due to non-positive step size after overlap calculation.")
            return False

        img_w, img_h = original_pil_img.size
        if img_w == 0 or img_h == 0: return False

        initial_offset_x = (img_w - tile_w) % step_w if img_w > tile_w and step_w > 0 else 0
        initial_offset_y = (img_h - tile_h) % step_h if img_h > tile_h and step_w > 0 else 0

        if img_w <= tile_w: initial_offset_x = 0
        if img_h <= tile_h: initial_offset_y = 0

        tile_counter = 0

        for y_start in range(initial_offset_y, img_h, step_h):
            if y_start >= img_h and img_h > 0 : continue

            for x_start in range(initial_offset_x, img_w, step_w):
                if x_start >= img_w and img_w > 0 : continue

                current_tile_orig_x1 = x_start
                current_tile_orig_y1 = y_start
                current_tile_orig_x2 = min(x_start + tile_w, img_w)
                current_tile_orig_y2 = min(y_start + tile_h, img_h)

                actual_crop_w = current_tile_orig_x2 - current_tile_orig_x1
                actual_crop_h = current_tile_orig_y2 - current_tile_orig_y1

                if actual_crop_w <= 0 or actual_crop_h <= 0: continue

                # Calculate padding for this specific tile before proceeding further
                total_tile_area = float(tile_w * tile_h)
                actual_content_area = float(actual_crop_w * actual_crop_h)

                if total_tile_area == 0: continue # Should be caught by earlier tile_w/h check

                current_padding_ratio = 1.0 - (actual_content_area / total_tile_area)

                if current_padding_ratio > max_padding_ratio_allowed:
                    continue

                actual_cropped_tile_pil = original_pil_img.crop((current_tile_orig_x1, current_tile_orig_y1,
                                                                current_tile_orig_x2, current_tile_orig_y2))

                final_tile_pil = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
                final_tile_pil.paste(actual_cropped_tile_pil, (0, 0))

                tile_boxes_relative_to_tile = []
                for box_label, box_x1_orig, box_y1_orig, box_x2_orig, box_y2_orig in original_boxes:
                    b_x1_o_abs, b_x2_o_abs = min(box_x1_orig, box_x2_orig), max(box_x1_orig, box_x2_orig)
                    b_y1_o_abs, b_y2_o_abs = min(box_y1_orig, box_y2_orig), max(box_y1_orig, box_y2_orig)

                    if (b_x1_o_abs >= current_tile_orig_x1 and
                        b_y1_o_abs >= current_tile_orig_y1 and
                        b_x2_o_abs <= current_tile_orig_x2 and
                        b_y2_o_abs <= current_tile_orig_y2):

                        adj_box_x1 = b_x1_o_abs - current_tile_orig_x1
                        adj_box_y1 = b_y1_o_abs - current_tile_orig_y1
                        adj_box_x2 = b_x2_o_abs - current_tile_orig_x1
                        adj_box_y2 = b_y2_o_abs - current_tile_orig_y1

                        if (adj_box_x2 - adj_box_x1) >= self.min_box_size and \
                           (adj_box_y2 - adj_box_y1) >= self.min_box_size:
                            tile_boxes_relative_to_tile.append((box_label, adj_box_x1, adj_box_y1, adj_box_x2, adj_box_y2))

                if tile_boxes_relative_to_tile:
                    tiles_generated_with_boxes = True
                    tile_counter += 1
                    tile_stem = f"{original_img_path.stem}_tile_{tile_counter}"
                    tile_img_name = f"{tile_stem}.png"
                    tile_label_name = f"{tile_stem}.txt"

                    try:
                        final_tile_pil.save(self.images_out_dir / tile_img_name)
                    except Exception as e:
                        print(f"Error saving tile image {tile_img_name}: {e}")
                        continue

                    tile_label_path = self.labels_dir / tile_label_name
                    with open(tile_label_path, 'w') as f_tile_label:
                        for lbl_name, rel_x1, rel_y1, rel_x2, rel_y2 in tile_boxes_relative_to_tile:
                            class_idx = self.label_to_idx.get(lbl_name)
                            if class_idx is None: continue

                            norm_rel_x1, norm_rel_x2 = min(rel_x1, rel_x2), max(rel_x1, rel_x2)
                            norm_rel_y1, norm_rel_y2 = min(rel_y1, rel_y2), max(rel_y1, rel_y2)

                            cx = ((norm_rel_x1 + norm_rel_x2) / 2) / tile_w
                            cy = ((norm_rel_y1 + norm_rel_y2) / 2) / tile_h
                            bw = (norm_rel_x2 - norm_rel_x1) / tile_w
                            bh = (norm_rel_y2 - norm_rel_y1) / tile_h

                            cx = max(0.0, min(1.0, cx)); cy = max(0.0, min(1.0, cy))
                            bw = max(0.0, min(1.0, bw)); bh = max(0.0, min(1.0, bh))

                            f_tile_label.write(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        return tiles_generated_with_boxes


if __name__ == '__main__':
    img_dir_standalone = PROJECT_ROOT / 'data' / 'images_unlabeled'
    labels_txt_standalone = PROJECT_ROOT / 'data' / 'labels.txt'
    labels_dir_standalone = PROJECT_ROOT / 'data' / 'labels'
    images_out_standalone = PROJECT_ROOT / 'data' / 'images'
    dataset_out_standalone = PROJECT_ROOT / 'dataset'

    for directory in [img_dir_standalone, labels_dir_standalone, images_out_standalone, dataset_out_standalone.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    if not labels_txt_standalone.exists():
        print(f"Warning: '{labels_txt_standalone.name}' not found. Creating dummy.")
        try:
            with open(labels_txt_standalone, 'w') as f_dummy: f_dummy.write("0:label1\n1:label2\n")
            print(f"Created dummy '{labels_txt_standalone.name}'. Please edit.")
        except Exception as e: print(f"Could not create dummy: {e}")

    if not any(f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] for f in img_dir_standalone.iterdir()):
         print(f"Warning: Directory '{img_dir_standalone.name}' has no images. Add images to {img_dir_standalone}.")

    try:
        tool = AnnotationTool(
            images_dir=img_dir_standalone,
            labels_txt=labels_txt_standalone,
            labels_dir=labels_dir_standalone
        )
        ok = tool.run()

        if ok:
            if any(images_out_standalone.iterdir()) and any(labels_dir_standalone.iterdir()):
                 print("Annotations completed. Splitting dataset...")
                 split_dataset(images_out_standalone, labels_dir_standalone, dataset_out_standalone)
            else:
                print("Annotation session completed, but no labeled images or labels found to split.")
        else:
            print("Annotation aborted by user.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
