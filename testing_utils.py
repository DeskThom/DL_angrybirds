import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_yolo_annotations(image_path, label_path, class_names=None):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    # Create a plot
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Read the YOLO label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Parse each line in the label file
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # skip incomplete lines
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # Convert normalized coordinates to absolute pixel values
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Calculate the top-left corner of the bounding box
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)

        # Create a rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # Add class label (if provided)
        if class_names and class_id < len(class_names):
            ax.text(
                x_min, max(y_min - 5, 0), class_names[class_id],
                color='red', fontsize=12, backgroundcolor='white'
            )

    ax.axis('off')
    plt.tight_layout()
    plt.show(block=True)  # <- ensures the window stays open