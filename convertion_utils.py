import json
import os
from PIL import Image

# The functions in this utils file are designed to convert the scarecrow dataset to both coco and yolo formats.
# To convert the entire dataset, please use process_all_subdirs(root_dir) function.

def convert_to_coco(your_data, images_folder):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "Bird"}]
    }
    annotation_id = 1

    for img_id, item in enumerate(your_data, start=1):
        filename = item["OriginalFileName"]
        img_path = os.path.join(images_folder, filename)
        
        # Read image size
        with Image.open(img_path) as img:
            width, height = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
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

# Note: Currently WIP and not working
def convert_to_coco_from_path(file_path, images_folder):
    # Get directory of the file
    folder = os.path.dirname(file_path)
    output_file = os.path.join(folder, 'coco_annotations.json')

    # Load your annotations.json file
    with open(file_path, 'r') as f:
        your_data = json.load(f)

    # Convert to COCO format
    coco_data = convert_to_coco(your_data, images_folder)

    # Save coco_annotations.json in the same folder
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f'✅ Converted to COCO format and saved as {output_file}')

def coco_to_yolo(coco_json_path, output_dir):
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_to_index = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}

    # Create output dir if needed
    os.makedirs(output_dir, exist_ok=True)

    # Group annotations by image_id
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_annotations.setdefault(ann['image_id'], []).append(ann)

    # Process each image
    for image_id, img in images.items():
        img_width = img['width']
        img_height = img['height']
        img_filename = os.path.splitext(img['file_name'])[0]
        txt_path = os.path.join(output_dir, f"{img_filename}.txt")

        with open(txt_path, 'w') as txt_file:
            for ann in image_annotations.get(image_id, []):
                cat_id = ann['category_id']
                yolo_cat_id = category_to_index[cat_id]
                x, y, w, h = ann['bbox']

                # Convert to YOLO normalized format
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                txt_file.write(f"{yolo_cat_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"✅ YOLO labels saved to '{output_dir}'")
    
def process_all_subdirs(root_dir):
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root_dir, split)
        ann_path = os.path.join(split_dir, 'annotations.json')

        if not os.path.exists(ann_path):
            print(f"❌ No annotations.json found in {split_dir}, skipping...")
            continue

        # Step 1: Convert to COCO
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        image_path = os.path.join(split_dir, 'images')
        if not os.path.exists(image_path):
            print(f"❌ No images folder found in {split_dir}, skipping...")
            continue
        
        coco_data = convert_to_coco(data, image_path)
        coco_out_path = os.path.join(split_dir, 'coco_annotations.json')
        with open(coco_out_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        print(f"✅ Saved COCO JSON to {coco_out_path}")

        # Step 2: Convert COCO to YOLO
        labels_out_dir = os.path.join(split_dir, 'labels')
        coco_to_yolo(coco_out_path, labels_out_dir)
        print(f"✅ Saved YOLO labels to {labels_out_dir}")