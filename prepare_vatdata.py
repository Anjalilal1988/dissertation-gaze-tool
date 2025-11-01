import json
import random
import os

# Path to the VAT dataset folder (update this!)
vat_path = r'C:\Users\anjal\OneDrive\Desktop\Thesis\VAT'  # Adjust to your actual path

# Function to parse .txt annotations with debugging
def parse_txt_annotations(file_path):
    annotations = []
    print(f"Processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments or empty lines
                parts = line.split(',')
                print(f"Line {line_idx}: {parts}")  # Debug: Print each line's parts
                if len(parts) == 7:  # Expecting image, frame, x1,y1,x2,y2, gaze_x (or -1)
                    try:
                        image = parts[0].strip()
                        # Only append .jpg if not already present
                        if not image.lower().endswith('.jpg'):
                            image += '.jpg'
                        print(f"Processed image name: {image} from original {parts[0].strip()}")
                        frame = int(parts[1])
                        x1, y1, x2, y2 = map(float, parts[2:6])  # Bounding box corners
                        # Convert x1,y1,x2,y2 to x,y,w,h
                        bbox = [x1, y1, x2 - x1, y2 - y1]  # Width and height
                        gaze_x = float(parts[6])
                        # Use gaze_x if valid, otherwise set to None; assume gaze_y missing
                        gaze = [gaze_x, -1.0] if gaze_x >= 0 else [-1.0, -1.0]
                        # Eye position placeholder (derive from bbox center)
                        eye = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
                        annotations.append({
                            'path': image,
                            'frame': frame,
                            'bbox': bbox,  # [x, y, w, h] in pixels (to be normalized later if needed)
                            'eye': eye,    # Placeholder
                            'gaze': gaze,  # [x, y] in pixels (to be normalized later if needed)
                            'type': 'image'
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line {line_idx}: {e} - Skipping line: {line}")
                else:
                    print(f"Line {line_idx} has incorrect parts: {len(parts)} - Skipping: {line}")
    return annotations

# Recursively find and parse all .txt files in annotations/train
annotations = []
for root, dirs, files in os.walk(os.path.join(vat_path, 'annotations', 'train')):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            annotations.extend(parse_txt_annotations(file_path))

print(f"Total annotations loaded: {len(annotations)} from {len(files)} .txt files.")

# Select 250 random diverse entries
if annotations:
    random.shuffle(annotations)
    selected_vat = annotations[:250]
else:
    print("No annotations found. Check file format or path.")
    selected_vat = []

# Output to JSON
output_path = os.path.abspath('vat_selected.json')
with open(output_path, 'w') as f:
    json.dump(selected_vat, f, indent=4)

print(f"Selected {len(selected_vat)} from VAT and saved to {output_path}")