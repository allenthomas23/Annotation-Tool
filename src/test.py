import tkinter as tk
import cv2
from PIL import Image, ImageTk

# --- Configuration ---
# Use the absolute path that we know works
IMAGE_PATH = '/Users/allenthomas/Code/Annotation-Tool/data/images_unlabeled/000000007873.jpg'
# ---------------------

try:
    # 1. Create the main window
    root = tk.Tk()
    root.title("Tkinter Image Display Test")

    # 2. Load the image with OpenCV (we know this part works)
    print("Loading image with cv2...")
    cv_image = cv2.imread(IMAGE_PATH)
    if cv_image is None:
        raise ValueError("cv2.imread failed, which is unexpected.")
    print("cv2 image loaded successfully.")

    # 3. Convert from OpenCV's BGR to RGB color format
    print("Converting color from BGR to RGB...")
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    print("Color converted.")

    # 4. Convert the image array to a Pillow Image object
    print("Converting to Pillow Image...")
    pil_image = Image.fromarray(rgb_image)
    print("Pillow Image created.")

    # 5. Convert the Pillow Image to a Tkinter PhotoImage object
    print("Converting to Tkinter PhotoImage...")
    photo_image = ImageTk.PhotoImage(image=pil_image)
    print("PhotoImage created.")

    # 6. Create a Canvas and display the image
    canvas = tk.Canvas(root, width=photo_image.width(), height=photo_image.height())
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
    print("\nImage should now be visible in the new window.")

    # This is a CRITICAL step in Tkinter.
    # You must keep a reference to the PhotoImage object,
    # otherwise it gets garbage-collected and the image disappears.
    # Attaching it to a widget (like the root window) is a common way to do this.
    root.image = photo_image

    # 7. Start the Tkinter event loop
    root.mainloop()

except Exception as e:
    print(f"\nAn error occurred: {e}")