import os
import sys
import json
from glob import glob


def load_annotations(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return []


def normalize_entry(entry):
    # Ensure expected structure; tolerate extra keys
    idx = entry.get('index')
    anns = entry.get('annotations', [])
    if not isinstance(anns, list):
        anns = []
    return {'index': idx, 'annotations': anns}


def merge_inputs(input_files, add_annotator=False):
    by_image = {}

    for path in input_files:
        src = os.path.basename(path)
        annotator_id = os.path.splitext(src)[0] if add_annotator else None
        data = load_annotations(path)
        if not isinstance(data, list):
            print(f"Skipping non-list file: {path}")
            continue
        for entry in data:
            entry = normalize_entry(entry)
            for ann in entry['annotations']:
                if not isinstance(ann, dict):
                    continue
                image_path = ann.get('image_path')
                if not image_path:
                    # Fallback: no image_path – group under special key
                    image_path = '__unknown__'
                # Optionally tag each annotation with annotator id
                if annotator_id and 'annotator_id' not in ann:
                    ann['annotator_id'] = annotator_id
                by_image.setdefault(image_path, []).append(ann)

    # Build consolidated list with fresh sequential indices
    merged = []
    next_index = 0
    for image_path, annotations in by_image.items():
        merged.append({
            'index': next_index,
            'annotations': annotations,
            'image_path': image_path  # helpful metadata; app ignores extra keys
        })
        next_index += 1
    return merged


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_annotations.py <input_dir_or_glob> [output_file] [--tag-annotator]")
        print("Examples:")
        print("  python merge_annotations.py participants/*.json annotations_merged.json --tag-annotator")
        print("  python merge_annotations.py participants annotations_merged.json")
        return 1

    input_arg = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].startswith('--') else 'annotations_merged.json'
    tag_annotator = any(a == '--tag-annotator' for a in sys.argv[1:])

    if os.path.isdir(input_arg):
        input_files = sorted(glob(os.path.join(input_arg, '*.json')))
    else:
        input_files = sorted(glob(input_arg))

    if not input_files:
        print(f"No input JSON files found for: {input_arg}")
        return 1

    merged = merge_inputs(input_files, add_annotator=tag_annotator)
    try:
        with open(output_file, 'w') as f:
            json.dump(merged, f)
        print(f"Wrote merged annotations to {output_file} (from {len(input_files)} files, {len(merged)} images)")
    except Exception as e:
        print(f"Failed to write merged file {output_file}: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())