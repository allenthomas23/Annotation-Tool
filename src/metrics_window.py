import tkinter as tk
from tkinter import messagebox
import multiprocessing
import queue
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MetricsWindow:
    def __init__(self, root: tk.Tk, update_q: multiprocessing.Queue):
        """Initialize the persistent metrics window for monitoring YOLO training runs."""
        self.root = root
        self.update_q = update_q
        self.root.title("Persistent Information Window")
        self.root.geometry("550x400")

        self.project_root = Path(__file__).resolve().parent
        self.yolo_runs_dir = self.project_root / "temp_training_artifacts" / "yolo_training_runs"
        self.processed_runs = set()
        self.progress_window = None 

        self.info_label_var = tk.StringVar(value="This window stays open. Monitoring training runs...")
        info_label = tk.Label(self.root, textvariable=self.info_label_var, padx=10, pady=10, wraplength=530, justify=tk.LEFT)
        info_label.pack(pady=5, fill=tk.X)

        
        progress_button = tk.Button(self.root, text="Show Overall Model Progress", command=self.show_overall_progress)
        progress_button.pack(pady=5)

        separator = tk.Frame(self.root, height=2, bd=1, relief=tk.SUNKEN)
        separator.pack(fill=tk.X, padx=5, pady=5)

        self.metrics_title_var = tk.StringVar(value="Latest Training Metrics:")
        metrics_title_label = tk.Label(self.root, textvariable=self.metrics_title_var, padx=10, pady=5, font=("Arial", 10, "bold"), justify=tk.LEFT)
        metrics_title_label.pack(fill=tk.X)

        self.metrics_text_var = tk.StringVar(value="No new training metrics yet.")
        metrics_label = tk.Label(self.root, textvariable=self.metrics_text_var, padx=10, pady=10, wraplength=530, height=10, relief=tk.GROOVE, anchor='nw', justify=tk.LEFT)
        metrics_label.pack(pady=5, fill=tk.BOTH, expand=True)

        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_pos = (screen_width // 2) - (width // 2) + 300
        y_pos = (screen_height // 2) - (height // 2) + 150
        self.root.geometry(f'{width}x{height}+{x_pos}+{y_pos}')

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.periodic_update()

    def check_message_queue(self):
        """Check the message queue for exit signals or other messages."""
        try:
            message = self.update_q.get_nowait()
            if message == "EXIT_PERSISTENT_WINDOW":
                print("Persistent window: Received EXIT signal.")
                self._on_closing()
                return False
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Persistent window: Error checking message queue: {e}")
        return True

    def check_for_new_training_runs(self):
        """Check for new YOLO training runs and update the metrics display."""
        if not self.yolo_runs_dir.exists() or not self.yolo_runs_dir.is_dir():
            return

        try:
            run_dirs = sorted(
                [d for d in self.yolo_runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else -1
            )

            latest_run_displayed_this_cycle = False
            #so latest is first
            for run_dir in reversed(run_dirs):
                run_name = run_dir.name
                if run_name not in self.processed_runs:
                    results_csv_path = run_dir / "results.csv"
                    if results_csv_path.exists() and results_csv_path.is_file():
                        self.display_metrics_from_csv(results_csv_path, run_name)
                        self.processed_runs.add(run_name)
                        latest_run_displayed_this_cycle = True
                        break 
            #If no new runs were displayed, check the last run again
            if not latest_run_displayed_this_cycle and run_dirs:
                last_run_dir = run_dirs[-1]
                results_csv_path = last_run_dir / "results.csv"
                current_metrics_title = self.metrics_title_var.get()
                if results_csv_path.exists() and results_csv_path.is_file() and last_run_dir.name not in current_metrics_title:
                    self.display_metrics_from_csv(results_csv_path, last_run_dir.name)
        except Exception as e:
            print(f"Persistent window: Error checking training runs: {e}")
            self.metrics_text_var.set(f"Error accessing training runs:\n{e}")

    def display_metrics_from_csv(self, csv_path: Path, run_name: str):
        """Read the latest metrics from a CSV file and update the display."""
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                self.metrics_text_var.set(f"Run: {run_name}\nNo data rows in CSV.")
                return
            
            last_row = df.iloc[-1]
            
            epoch = last_row.get('epoch', 'N/A')
            map50 = last_row.get('metrics/mAP50(B)', 'N/A')
            map50_95 = last_row.get('metrics/mAP50-95(B)', 'N/A')
            val_box_loss = last_row.get('val/box_loss', 'N/A')
            val_cls_loss = last_row.get('val/cls_loss', 'N/A')
            train_box_loss = last_row.get('train/box_loss', 'N/A')
            train_cls_loss = last_row.get('train/cls_loss', 'N/A')

            def format_metric(value):
                try: return f"{float(value):.4f}"
                except (ValueError, TypeError): return 'N/A'

            display_text = (
                f"Run: {run_name} (Epoch: {epoch})\n"
                f"------------------------------------\n"
                f"Validation Metrics:\n"
                f"  mAP@0.50 (B):     {format_metric(map50)}\n"
                f"  mAP@0.50-0.95 (B): {format_metric(map50_95)}\n"
                f"  Box Loss:         {format_metric(val_box_loss)}\n"
                f"  Class Loss:       {format_metric(val_cls_loss)}\n"
                f"Training Losses:\n"
                f"  Box Loss:         {format_metric(train_box_loss)}\n"
                f"  Class Loss:       {format_metric(train_cls_loss)}\n"
            )
            self.metrics_title_var.set(f"Latest Training Metrics (from {run_name}):")
            self.metrics_text_var.set(display_text)

        except FileNotFoundError:
            self.metrics_text_var.set(f"Run: {run_name}\nError: results.csv not found.")
        except Exception as e:
            print(f"Persistent window: Error reading CSV {csv_path} for run {run_name}: {e}")
            self.metrics_text_var.set(f"Run: {run_name}\nError reading metrics CSV:\n{e}")

    def periodic_update(self):
        """Periodically check for new training runs and update the metrics display."""
        if not self.check_message_queue():
            return
        self.check_for_new_training_runs()
        if hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
            self.root.after(3000, self.periodic_update)

    def show_overall_progress(self):
        """Display a window showing overall model progress across all training runs."""
        print("Persistent window: 'Show Overall Model Progress' clicked.")
        if not self.yolo_runs_dir.exists() or not self.yolo_runs_dir.is_dir():
            messagebox.showinfo("No Data", "Training artifacts directory not found.", parent=self.root)
            return

        all_runs_data = []
        try:
            run_dirs = sorted(
                [d for d in self.yolo_runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else -1
            )

            for run_dir in run_dirs:
                results_csv_path = run_dir / "results.csv"
                if results_csv_path.exists() and results_csv_path.is_file():
                    df = pd.read_csv(results_csv_path)
                    if not df.empty:
                        last_row = df.iloc[-1]
                        run_metrics = {
                            'run': int(run_dir.name.split('_')[-1]),
                            'epoch': last_row.get('epoch', 0),
                            'mAP50(B)': pd.to_numeric(last_row.get('metrics/mAP50(B)'), errors='coerce'),
                            'mAP50-95(B)': pd.to_numeric(last_row.get('metrics/mAP50-95(B)'), errors='coerce'),
                            'val_box_loss': pd.to_numeric(last_row.get('val/box_loss'), errors='coerce'),
                            'val_cls_loss': pd.to_numeric(last_row.get('val/cls_loss'), errors='coerce')
                        }
                        all_runs_data.append(run_metrics)
        except Exception as e:
            messagebox.showerror("Error Reading Data", f"Could not read all run data: {e}", parent=self.root)
            print(f"Error gathering overall progress: {e}")
            return

        if not all_runs_data:
            messagebox.showinfo("No Data", "No completed training run data found to display.", parent=self.root)
            return

        #If a progress window already exists, destroy it before creating a new one
        if self.progress_window and self.progress_window.winfo_exists():
            self.progress_window.destroy()

        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Overall Model Progress")
        self.progress_window.geometry("800x600")

        df_runs = pd.DataFrame(all_runs_data)

        fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True) # Two subplots
        fig.suptitle("Model Performance Over Runs", fontsize=14)

        # Plot mAP scores
        axes[0].plot(df_runs['run'], df_runs['mAP50(B)'], marker='o', linestyle='-', label='mAP@0.50 (B)')
        axes[0].plot(df_runs['run'], df_runs['mAP50-95(B)'], marker='s', linestyle='--', label='mAP@0.50-0.95 (B)')
        axes[0].set_ylabel('mAP Scores')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title('Validation mAP Scores')

        # Plot validation losses
        axes[1].plot(df_runs['run'], df_runs['val_box_loss'], marker='o', linestyle='-', label='Validation Box Loss')
        axes[1].plot(df_runs['run'], df_runs['val_cls_loss'], marker='s', linestyle='--', label='Validation Class Loss')
        axes[1].set_xlabel('Training Run Number')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_title('Validation Losses')
        
        plt.xticks(df_runs['run']) # Ensure x-axis shows integer run numbers

        canvas = FigureCanvasTkAgg(fig, master=self.progress_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

    def _on_closing(self):
        """Handle window closing event."""
        print("Persistent window: Closing.")
        if self.progress_window and self.progress_window.winfo_exists():
            self.progress_window.destroy()
        if hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
            try:
                self.root.quit()
                self.root.destroy()
            except tk.TclError:
                pass

def metrics_window_process(update_q: multiprocessing.Queue):
    """Start the persistent metrics window process."""
    try:
        root = tk.Tk()
        app = MetricsWindow(root, update_q)
        root.mainloop()
    except Exception as e:
        print(f"FATAL error starting persistent window process: {e}")
        import traceback
        traceback.print_exc()
