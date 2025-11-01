import json
import random
import os

# Paths to annotation files
train_annotations_file = r'C:\Users\anjal\OneDrive\Desktop\Thesis\Gazefollow\train_annotations_release.txt'
test_annotations_file = r'C:\Users\anjal\OneDrive\Desktop\Thesis\Gazefollow\test_annotations_release.txt'

# Function to parse annotations
def parse_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 16:
                    path = parts[0]
                    left = float(parts[2])
                    top = float(parts[3])
                    right = float(parts[4])
                    bottom = float(parts[5])
                    if top > bottom:
                        top, bottom = bottom, top
                    bbox = [left, top, right - left, bottom - top]
                    eye = [float(parts[6]), float(parts[7])]
                    gaze = [float(parts[8]), float(parts[9])]
                    annotations.append({
                        'path': path,
                        'bbox': bbox,
                        'eye': eye,
                        'gaze': gaze,
                        'type': 'image'
                    })
    return annotations

# Load all annotations
all_annotations = parse_annotations(train_annotations_file) + parse_annotations(test_annotations_file)

# Select 250 random
random.shuffle(all_annotations)
selected_gazefollow = all_annotations[:250]

# Output to JSON
output_path = os.path.abspath('gazefollow_selected.json')
with open(output_path, 'w') as f:
    json.dump(selected_gazefollow, f, indent=4)

print(f"Selected {len(selected_gazefollow)} from GazeFollow and saved to {output_path}")
