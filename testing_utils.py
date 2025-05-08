import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml

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
    
    
# GENERATE YAML
def find_dataset_folder(folder_name="scarecrow_dataset"):
    """
    Finds the dataset folder in the current working directory or its subdirectories and returns it
    """
    # Get the current working directory
    current_working_dir = os.getcwd()
    print(f"Current working directory: {current_working_dir}")
    
    # Check if the folder exists in the current working directory
    possible_path = os.path.join(current_working_dir, folder_name)
    if os.path.isdir(possible_path):
        print(f"Found '{folder_name}' in the current directory: {possible_path}")
        return possible_path  # If the folder is found, return the path
    
    # If not found in the current working directory, start walking through the directory tree
    print(f"'{folder_name}' not found in the current working directory. Searching...")
    for root, dirs, files in os.walk(current_working_dir):
        if folder_name in dirs:
            found_path = os.path.join(root, folder_name)
            print(f"Found '{folder_name}' at: {found_path}")
            return found_path  # Return the full path if found
    
    # If folder isn't found
    print(f"'{folder_name}' not found after searching the entire directory.")
    return None  # Return None if the folder is not found

def generate_yaml(folder_name="scarecrow_dataset", yaml_filename="data.yaml"):
    """
    Automatically generates a YAML file for using the YOLO model 
    """
    # Find the folder path
    dataset_base_path = find_dataset_folder(folder_name)

    if dataset_base_path:
        # Create the dictionary for the YAML file with the correct structure
        data = {
            'path': dataset_base_path,  # dataset root dir
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # Number of classes
            'names': ['Bird']  # Class names
        }

        # Write the data to a YAML file with the proper structure
        with open(yaml_filename, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False, indent = 2)

        print(f"YAML file written as {yaml_filename} with path: {dataset_base_path}")
    else:
        print(f"Dataset folder '{folder_name}' not found in {os.getcwd()}")
