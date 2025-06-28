import cv2
from pathlib import Path

# --- Configuration ---
IMAGE_NAME = "00_tile_1.png"  # Replace with your image file name
LABEL_FILE_NAME = "00_tile_1.txt" # Replace with your corresponding label file name
# ---------------------

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"

image_path = IMAGES_DIR / IMAGE_NAME
label_path = LABELS_DIR / LABEL_FILE_NAME

# Load the image
image = cv2.imread(str(image_path))
if image is None:
    print(f"Error: Could not load image {image_path}")
    exit()

img_h, img_w = image.shape[:2]

# Read labels and draw boxes
if label_path.exists():
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # class_id, center_x, center_y, width, height
            # class_id = int(parts[0]) # Not used in this minimal script
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            # Denormalize coordinates
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box, thickness 2
else:
    print(f"Warning: Label file not found at {label_path}")

# Display the image
cv2.imshow("Image with Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Displayed image: {IMAGE_NAME}")