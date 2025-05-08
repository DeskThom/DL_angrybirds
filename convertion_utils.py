import json
import os
from PIL import Image


#this changes the annations format to Txt which Yolo can read. 
def convert_to_yolo_format(data_path, annotations_file, output_dir):
    # Load annotations JSON file
    with open(annotations_file, 'r') as f:
        annotations_list = json.load(f)

    # Ensure output directories exist
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    #dit moest blijkbaar
    class_map = {"Bird": 0}

    # ocess each image
    for entry in annotations_list:
        image_name = entry['OriginalFileName']
        annotation_data = entry['AnnotationData']


        # Load image to get width and height
        image_path = os.path.join(data_path, 'images', image_name)
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            #print(f"Image size: {img_width} x {img_height}")

        # Create the label file for this image
        label_file = os.path.join(output_dir, 'labels', os.path.splitext(image_name)[0] + '.txt')

        with open(label_file, 'w') as label_f:
            for obj in annotation_data:
                class_name = obj['Label']  # 'Label' field in your data
                
                if class_name in class_map:
                    # Get the coordinates (bounding box)
                    coordinates = obj['Coordinates']
                    
                    # Calculate bounding box (x_min, y_min, width, height)
                    x_min = min([coord['X'] for coord in coordinates])
                    y_min = min([coord['Y'] for coord in coordinates])
                    x_max = max([coord['X'] for coord in coordinates])
                    y_max = max([coord['Y'] for coord in coordinates])

                    # YOLO format: class_id x_center y_center width height (all normalized)
                    class_id = class_map[class_name]
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    norm_width = (x_max - x_min) / img_width
                    norm_height = (y_max - y_min) / img_height
                    
                    # Write the YOLO annotation to the label file
                    label_f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")
                    
           
        print("Created txt for:"+image_name)
        

def convert_to_coco(your_data):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "Bird"}]
    }
    annotation_id = 1

    for img_id, item in enumerate(your_data, start=1):
        filename = item["OriginalFileName"]
        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": 640,   # Adjust if known
            "height": 480   # Adjust if known
        })

        for ann in item["AnnotationData"]:
            coords = ann["Coordinates"]
            x_coords = [p["X"] for p in coords]
            y_coords = [p["Y"] for p in coords]
            x_min, y_min = min(x_coords), min(y_coords)
            width, height = max(x_coords) - x_min, max(y_coords) - y_min

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_min, y_min, width, height],
                "area": width * height
            })
            annotation_id += 1

    return coco


def main(file_path):
    # Get directory of the file
    folder = os.path.dirname(file_path)
    output_file = os.path.join(folder, 'coco_annotations.json')

    # Load your annotations.json file
    with open(file_path, 'r') as f:
        your_data = json.load(f)

    # Convert to COCO format
    coco_data = convert_to_coco(your_data)

    # Save coco_annotations.json in the same folder
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f'✅ Converted to COCO format and saved as {output_file}')

# Example usage:
# main('path/to/annotations.json')