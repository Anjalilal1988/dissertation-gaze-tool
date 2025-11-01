from flask import Flask, request, redirect, url_for, render_template_string, send_file, abort, session, jsonify
import json
import random
import os
from PIL import Image
import time
import portalocker

app = Flask(__name__)
app.secret_key = 'your-secret-key-for-sessions'  # Required for sessions

# Avoid storing large objects in the cookie-based session.
# Keep per-user image assignments in server memory keyed by a tiny uid.
USER_IMAGE_ASSIGNMENTS = {}

# Configuration for images per user
IMAGES_PER_USER = 500  # Fixed set: show same 500 images to all users

# Load JSON file with all available images
json_path = os.path.abspath('combined_gazefollow_vat.json')
try:
    with open(json_path, 'r') as f:
        all_images = json.load(f)
    print(f"Loaded {len(all_images)} total entries from {json_path}")
    if not all_images:
        print("Warning: JSON is empty or no entries loaded.")
except Exception as e:
    print(f"Error loading JSON: {e}")
    all_images = []

# Merged images base (serve only these 500 images)
project_root = os.path.abspath(os.path.dirname(__file__))
# Allow overriding merged images root via environment (useful for deployment with mounted volumes)
merged_root = os.environ.get('MERGED_ROOT', os.path.join(project_root, 'merged_images'))
merged_gf_root = os.path.join(merged_root, 'gazefollow')
merged_vat_root = os.path.join(merged_root, 'vat')

# Classify focal point region for a normalized gaze (gx, gy)
def classify_focal_point_norm(gx, gy):
    try:
        gx = float(gx)
        gy = float(gy)
    except Exception:
        return None

    # Out-of-frame handling (normalized coordinates outside [0,1])
    if gx < 0 or gx > 1 or gy < 0 or gy > 1:
        horiz = 'left' if gx < 0 else ('right' if gx > 1 else 'center')
        vert = 'top' if gy < 0 else ('bottom' if gy > 1 else 'middle')
        if horiz == 'center' and vert == 'middle':
            return 'out-of-frame'
        if vert == 'middle':
            return f'out-of-frame {horiz}'
        if horiz == 'center':
            return f'out-of-frame {vert}'
        return f'out-of-frame {vert}-{horiz}'

    # In-frame: 3x3 grid classification
    horiz = 'left' if gx < (1.0/3.0) else ('right' if gx > (2.0/3.0) else 'center')
    vert = 'top' if gy < (1.0/3.0) else ('bottom' if gy > (2.0/3.0) else 'middle')
    if horiz == 'center' and vert == 'middle':
        return 'center'
    if vert == 'middle':
        return horiz
    if horiz == 'center':
        return vert
    return f'{vert}-{horiz}'

def collect_merged_sets():
    """Collect available merged image identifiers for filtering.
    - For GazeFollow: relative paths like 'train/..../...jpg' or 'test2/...'
    - For VAT: basenames (filenames) available under merged_vat_root
    """
    gf_rel_paths = set()
    vat_basenames = set()
    try:
        for root, dirs, files in os.walk(merged_gf_root):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), merged_gf_root)
                gf_rel_paths.add(rel.replace(os.sep, '/'))
    except Exception as e:
        print(f"Warning: failed to scan merged GazeFollow: {e}")
    try:
        for root, dirs, files in os.walk(merged_vat_root):
            for f in files:
                vat_basenames.add(f)
    except Exception as e:
        print(f"Warning: failed to scan merged VAT: {e}")
    return gf_rel_paths, vat_basenames

gf_set, vat_set = collect_merged_sets()
available_images = []
if all_images:
    for item in all_images:
        p = item.get('path', '')
        if isinstance(p, str) and (p.startswith('train/') or p.startswith('test2/')):
            if p in gf_set:
                available_images.append(item)
        else:
            fname = os.path.basename(p)
            if fname in vat_set:
                available_images.append(item)
    print(f"Filtered to {len(available_images)} images available in merged_images (GF set={len(gf_set)}, VAT set={len(vat_set)})")
else:
    print("No all_images loaded; available_images remains empty.")

# JSON file for storing annotations (configurable via env)
# Set ANNOTATIONS_PATH to a persistent location in production, e.g., /var/data/annotations.json
annotations_file = os.environ.get('ANNOTATIONS_PATH', os.path.join(project_root, 'annotations.json'))

# Ensure directory exists if a nested path is used
annotations_dir = os.path.dirname(annotations_file)
if annotations_dir and not os.path.exists(annotations_dir):
    try:
        os.makedirs(annotations_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create annotations directory '{annotations_dir}': {e}")

# Initialize file if missing
if not os.path.exists(annotations_file):
    try:
        with open(annotations_file, 'w') as f:
            json.dump([], f)
    except Exception as e:
        print(f"Warning: could not initialize annotations file '{annotations_file}': {e}")

def get_next_available_index():
    """Get the next available index for annotations.json under a lock."""
    try:
        with portalocker.Lock(annotations_file, 'r+', timeout=10) as f:
            f.seek(0)
            try:
                data = json.load(f)
            except Exception:
                data = []
            max_index = -1
            for entry in data:
                if 'index' in entry:
                    try:
                        max_index = max(max_index, int(entry['index']))
                    except Exception:
                        pass
            return max_index + 1
    except Exception as e:
        print(f"Error getting next index: {e}")
        return 0

def reserve_annotation_index():
    """Reserve and return a unique annotation index using an exclusive file lock."""
    try:
        with portalocker.Lock(annotations_file, 'r+', timeout=10) as f:
            f.seek(0)
            try:
                data = json.load(f)
            except Exception:
                data = []
            max_index = -1
            for entry in data:
                if 'index' in entry:
                    try:
                        max_index = max(max_index, int(entry['index']))
                    except Exception:
                        pass
            new_index = max_index + 1
            data.append({'index': new_index, 'annotations': []})
            f.seek(0)
            f.truncate()
            json.dump(data, f)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
            return new_index
    except Exception as e:
        print(f"Error reserving next index: {e}")
        return get_next_available_index()

def _get_or_create_uid():
    uid = session.get('uid')
    if not uid:
        uid = os.urandom(8).hex()
        session['uid'] = uid
    return uid

def get_user_images():
    """Return the same fixed set of merged images to all users."""
    uid = _get_or_create_uid()
    user_images = USER_IMAGE_ASSIGNMENTS.get(uid)
    if not user_images:
        # Serve the filtered merged set. If more than IMAGES_PER_USER, trim deterministically.
        if len(available_images) > IMAGES_PER_USER:
            user_images = available_images[:IMAGES_PER_USER]
        else:
            user_images = available_images.copy()
        USER_IMAGE_ASSIGNMENTS[uid] = user_images

        # Initialize user's annotation index mapping (kept small in session)
        if 'user_annotation_indices' not in session:
            session['user_annotation_indices'] = {}

        print(f"Assigned {len(user_images)} merged images to user uid={uid} (fixed set)")
    return USER_IMAGE_ASSIGNMENTS[uid]

# API to save 3D gaze (camera coordinates) for a given image index
@app.route('/api/save_gaze3d/<int:index>', methods=['POST'])
def save_gaze3d(index):
    try:
        user_images = get_user_images()
        if not user_images or index < 0 or index >= len(user_images):
            return jsonify({"error": "Index out of range or no data loaded"}), 400

        payload = request.get_json(silent=True) or {}
        if not all(k in payload for k in ("X", "Y", "Z")):
            return jsonify({"error": "Missing X/Y/Z in JSON body"}), 400

        try:
            X = float(payload["X"])
            Y = float(payload["Y"])
            Z = float(payload["Z"])
        except Exception:
            return jsonify({"error": "X/Y/Z must be numeric"}), 400

        # Optional annotation index within this image's annotations list
        ann_idx = payload.get("annotation_idx")
        try:
            ann_idx = int(ann_idx) if ann_idx is not None else None
        except Exception:
            ann_idx = None

        # Optional gaze_number metadata to store alongside
        gaze_number = payload.get("gaze_number")
        try:
            gaze_number = int(gaze_number) if gaze_number is not None else None
        except Exception:
            gaze_number = None

        # Ensure an annotation index exists for this user's image
        if 'user_annotation_indices' not in session:
            session['user_annotation_indices'] = {}
        if str(index) not in session['user_annotation_indices']:
            session['user_annotation_indices'][str(index)] = reserve_annotation_index()
        annotation_index = session['user_annotation_indices'][str(index)]

        # Load and update annotations.json with exclusive lock
        with portalocker.Lock(annotations_file, 'r+', timeout=10) as f:
            f.seek(0)
            try:
                all_annotations = json.load(f)
            except Exception:
                all_annotations = []

            existing_entry_idx = None
            for i, entry in enumerate(all_annotations):
                if entry.get('index') == annotation_index:
                    existing_entry_idx = i
                    break

            # Ensure there is an entry for this annotation index
            if existing_entry_idx is None:
                all_annotations.append({'index': annotation_index, 'annotations': []})
                existing_entry_idx = len(all_annotations) - 1

            entry = all_annotations[existing_entry_idx]
            if 'annotations' not in entry or not isinstance(entry['annotations'], list):
                entry['annotations'] = []

            # Decide which annotation object to update
            if ann_idx is not None and 0 <= ann_idx < len(entry['annotations']):
                target_ann = entry['annotations'][ann_idx]
                target_ann['gaze_3d'] = [X, Y, Z]
                if gaze_number is not None:
                    target_ann['gaze_number'] = gaze_number
                if 'image_path' not in target_ann:
                    target_ann['image_path'] = user_images[index].get('path')
            else:
                # If specific index not provided or out of range, append a minimal annotation
                new_ann = {'bbox': None, 'gaze': None, 'gaze_3d': [X, Y, Z], 'image_path': user_images[index].get('path')}
                if gaze_number is not None:
                    new_ann['gaze_number'] = gaze_number
                entry['annotations'].append(new_ann)

            f.seek(0)
            f.truncate()
            json.dump(all_annotations, f)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        return jsonify({"status": "ok", "index": annotation_index, "annotation_idx": ann_idx, "gaze_3d": [X, Y, Z]})
    except Exception as e:
        print(f"save_gaze3d error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/gaze_suggest/<int:index>', methods=['GET'])
def gaze_suggest(index):
    """Return a best-effort gaze suggestion for the given image index.
    Uses existing dataset metadata (bbox/eye/gaze) and normalizes to [0,1].
    For VAT entries with pixel coordinates, converts to normalized values.
    Negative bbox widths/heights are rectified to positive with adjusted origin.
    Out-of-frame gaze values are mapped to small negative normalized values to trigger UI handling.
    """
    try:
        user_images = get_user_images()
        if not user_images or index < 0 or index >= len(user_images):
            return jsonify({"error": "Index out of range or no data loaded"}), 400

        item = user_images[index]
        full_path = resolve_image_full_path(index)
        with Image.open(full_path) as img:
            width, height = img.size

        def rectify_bbox(x, y, w, h):
            # Ensure positive width/height; shift origin if needed
            if w is not None and h is not None:
                if w < 0:
                    x = x + w
                    w = -w
                if h < 0:
                    y = y + h
                    h = -h
            return x, y, w, h

        def clamp01(v):
            try:
                return max(0.0, min(1.0, float(v)))
            except Exception:
                return 0.0

        def normalize_bbox(bbox, is_normalized):
            if not bbox or len(bbox) < 4:
                return [0.25, 0.25, 0.5, 0.5]
            x, y, w, h = [float(v) for v in bbox[:4]]
            x, y, w, h = rectify_bbox(x, y, w, h)
            if not is_normalized:
                x /= width
                y /= height
                w /= width
                h /= height
            # Clamp to [0,1]
            return [clamp01(x), clamp01(y), clamp01(w), clamp01(h)]

        def normalize_point(pt, is_normalized):
            if not pt or len(pt) < 2:
                return [0.5, 0.5]
            px, py = pt[:2]
            try:
                px = float(px)
                py = float(py)
            except Exception:
                return [0.5, 0.5]
            if is_normalized:
                return [px, py]
            # VAT pixel coordinates: -1 indicates out-of-frame
            px_norm = (px / width) if px >= 0 else -0.05
            py_norm = (py / height) if py >= 0 else -0.05
            return [px_norm, py_norm]

        p = item.get('path', '')
        is_gf = isinstance(p, str) and (p.startswith('train/') or p.startswith('test2/'))
        is_norm = is_gf

        bbox = normalize_bbox(item.get('bbox'), is_norm)
        eye = normalize_point(item.get('eye'), is_norm)
        gaze = normalize_point(item.get('gaze') or eye, is_norm)

        return jsonify({
            "index": index,
            "bbox": bbox,
            "eye": eye,
            "gaze": gaze,
            "source": "dataset"
        })
    except Exception as e:
        print(f"gaze_suggest error: {e}")
        return jsonify({"error": str(e)}), 500

def find_image_file(base_dir, filename):
    """Recursively search for filename (with or without .jpg) in base_dir."""
    print(f"Starting fallback search for {filename} in {base_dir}")
    base_name = os.path.splitext(filename)[0]  # Strip .jpg if double extension
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == filename or os.path.splitext(file)[0] == base_name:
                full_path = os.path.join(root, file)
                print(f"Found image at fallback path: {full_path}")
                return full_path
    print(f"Failed to find {filename} or {base_name} in {base_dir}")
    return None

def resolve_image_full_path(index):
    """Resolve and verify full image path for a user image index from merged_images."""
    user_images = get_user_images()
    if not user_images or index < 0 or index >= len(user_images):
        raise ValueError(f"Invalid index or no data: index={index}, len(user_images)={len(user_images)}")

    item = user_images[index]
    filename = os.path.basename(item['path'])
    base_name = os.path.splitext(filename)[0]
    print(f"Resolving merged image path for index {index}: {item['path']}")

    if item['path'].startswith('test2/') or item['path'].startswith('train/'):
        # GazeFollow in merged set
        full_path = os.path.join(merged_gf_root, item['path'].replace('/', os.sep))
        print(f"Merged GazeFollow path attempt: {full_path}")
    else:
        # VAT in merged set: find by filename under merged_vat_root
        full_path = find_image_file(merged_vat_root, filename) or find_image_file(merged_vat_root, base_name + '.jpg')
        print(f"Merged VAT search result: {full_path}")

    if not full_path or not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found in merged set: {filename}")

    # Verify image integrity (optional)
    try:
        with Image.open(full_path) as img:
            print(f"Image verified: {full_path}, size={img.size}")
    except Exception as e:
        raise RuntimeError(f"Invalid image file: {full_path}, error: {e}")

    return full_path

# Route to serve images with detailed debugging (from merged_images)
@app.route('/images/<int:index>')
def serve_image(index):
    try:
        full_path = resolve_image_full_path(index)
    except Exception as e:
        print(f"serve_image error: {e}")
        abort(400, description=str(e))
    print(f"Serving image: {full_path}")
    return send_file(full_path, mimetype='image/jpeg')

# VGGT API endpoints removed

@app.route('/')
def home():
    user_images = get_user_images()
    
    if not user_images:
        print("No data loaded, returning error page.")
        return "No data loaded."

    # Redirect to the first image
    return redirect(url_for('label_image', index=0))

@app.route('/label_image/<int:index>', methods=['GET', 'POST'])
def label_image(index):
    user_images = get_user_images()
    
    if index < 0 or index >= len(user_images):
        print(f"Invalid index: {index}")
        return "Index out of range", 400

    if request.method == 'POST':
        annotations_data = request.form.get('annotations')
        if annotations_data:
            annotations = json.loads(annotations_data)

            # Attach image_path to each annotation
            item = get_user_images()[index]
            for ann in annotations:
                if isinstance(ann, dict):
                    ann['image_path'] = item.get('path')
                    # Add focal point classification if gaze present
                    gaze = ann.get('gaze')
                    if isinstance(gaze, (list, tuple)) and len(gaze) == 2:
                        fp = classify_focal_point_norm(gaze[0], gaze[1])
                        if fp is not None:
                            ann['focal_point'] = fp

            # Get or assign a unique annotation index for this user's image
            if 'user_annotation_indices' not in session:
                session['user_annotation_indices'] = {}
            if str(index) not in session['user_annotation_indices']:
                session['user_annotation_indices'][str(index)] = reserve_annotation_index()

            annotation_index = session['user_annotation_indices'][str(index)]

            # Update annotations.json using exclusive lock
            with portalocker.Lock(annotations_file, 'r+', timeout=10) as f:
                f.seek(0)
                try:
                    all_annotations = json.load(f)
                except Exception:
                    all_annotations = []

                # Check if this annotation index already exists and update it, otherwise append
                existing_entry_idx = None
                for i, entry in enumerate(all_annotations):
                    if entry.get('index') == annotation_index:
                        existing_entry_idx = i
                        break

                annotation_entry = {'index': annotation_index, 'annotations': annotations}

                if existing_entry_idx is not None:
                    all_annotations[existing_entry_idx] = annotation_entry
                    print(f"Updated existing annotations for index {annotation_index}: {annotations}")
                else:
                    all_annotations.append(annotation_entry)
                    print(f"Added new annotations for index {annotation_index}: {annotations}")

                f.seek(0)
                f.truncate()
                json.dump(all_annotations, f)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            
            next_index = min(index + 1, len(user_images) - 1)
            if next_index == index:  # We've reached the last image
                return f"<h2>Annotation Complete!</h2><p>You have completed annotating all {len(user_images)} images assigned to you.</p><p>Total annotations saved with automatic indexing.</p><a href='{url_for('home')}'>Start Over</a>"
            
            return redirect(url_for('label_image', index=next_index))

    # Get the image URL with cache busting query parameter
    timestamp = int(time.time())
    image_url = url_for('serve_image', index=index) + f'?t={timestamp}'
    
    # Show progress information
    progress_info = f"Image {index + 1} of {len(user_images)} (User Session)"

    html = """
<!DOCTYPE html>
<html>
<head>
<title>Image Annotation</title>
<style>
  .image-container { position: relative; display: inline-block; }
  .annotations { margin-top: 20px; }
  .annotation { border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
</style>
<script>
let annotations = [];
let currentAnnotation = null;
let isDrawingBBox = false;
let isDrawingGaze = false;
let startX, startY;
const currentIndex = {{ index }};

async function autoDetectGaze() {
  try {
    const res = await fetch(`/api/gaze_suggest/${currentIndex}`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    if (data.error) {
      throw new Error(data.error);
    }
    const suggested = {
      bbox: data.bbox,
      gaze: data.gaze
    };
    annotations.push(suggested);
    const canvas = document.getElementById('canvas');
    const image = document.getElementById('image');
    const ctx = canvas.getContext('2d');
    redrawCanvas(ctx, image, annotations);
    updateAnnotationsList();
  } catch (err) {
    alert(`Gaze suggestion failed: ${err.message}`);
    console.error(err);
  }
}

function initCanvas() {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const image = document.getElementById('image');
  canvas.width = image.clientWidth;
  canvas.height = image.clientHeight;

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    if (!isDrawingGaze) {
      isDrawingBBox = true;
      currentAnnotation = { bbox: [startX / canvas.width, startY / canvas.height, 0, 0], gaze: null };
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (isDrawingBBox) {
      const rect = canvas.getBoundingClientRect();
      const endX = e.clientX - rect.left;
      const endY = e.clientY - rect.top;
      redrawCanvas(ctx, image, annotations);
      drawBBox(ctx, startX, startY, endX - startX, endY - startY, 'green');
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    if (isDrawingBBox) {
      const rect = canvas.getBoundingClientRect();
      const endX = e.clientX - rect.left;
      const endY = e.clientY - rect.top;
      currentAnnotation.bbox[2] = (endX - startX) / canvas.width;
      currentAnnotation.bbox[3] = (endY - startY) / canvas.height;
      isDrawingBBox = false;
      isDrawingGaze = true;  // Now ready to draw gaze
    } else if (isDrawingGaze) {
      const rect = canvas.getBoundingClientRect();
      const endX = e.clientX - rect.left;
      const endY = e.clientY - rect.top;
      currentAnnotation.gaze = [endX / canvas.width, endY / canvas.height];
      annotations.push(currentAnnotation);
      currentAnnotation = null;
      isDrawingGaze = false;
      redrawCanvas(ctx, image, annotations);
      updateAnnotationsList();
    }
  });
}

function redrawCanvas(ctx, image, anns) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  anns.forEach((ann, index) => {
    const [x, y, w, h] = ann.bbox;
    drawBBox(ctx, x * ctx.canvas.width, y * ctx.canvas.height, w * ctx.canvas.width, h * ctx.canvas.height, 'green');
    
    // Draw gaze number label on the bounding box
    const gazeNumber = index + 1;
    drawText(ctx, `Gaze ${gazeNumber}`, x * ctx.canvas.width + 5, y * ctx.canvas.height - 5, 'white', 'green');
    
    if (ann.gaze) {
      const [gx, gy] = ann.gaze;
      const [bx, by, bw, bh] = ann.bbox;
      const centerX = (bx + bw / 2) * ctx.canvas.width;
      const centerY = (by + bh / 2) * ctx.canvas.height;
      drawLine(ctx, centerX, centerY, gx * ctx.canvas.width, gy * ctx.canvas.height, 'blue');
      drawGaze(ctx, gx * ctx.canvas.width, gy * ctx.canvas.height);
      
      // Draw gaze number label near the gaze point
      drawText(ctx, `${gazeNumber}`, gx * ctx.canvas.width + 10, gy * ctx.canvas.height - 10, 'white', 'blue');
    }
  });
  if (currentAnnotation && currentAnnotation.bbox) {
    const [x, y, w, h] = currentAnnotation.bbox;
    drawBBox(ctx, x * ctx.canvas.width, y * ctx.canvas.height, w * ctx.canvas.width, h * ctx.canvas.height, 'green');
    
    // Show preview number for current annotation being drawn
    const nextGazeNumber = anns.length + 1;
    drawText(ctx, `Gaze ${nextGazeNumber}`, x * ctx.canvas.width + 5, y * ctx.canvas.height - 5, 'white', 'orange');
  }
}

function drawBBox(ctx, x, y, w, h, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
}

function drawLine(ctx, x1, y1, x2, y2, color) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawGaze(ctx, x, y) {
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, 2 * Math.PI);
  ctx.fillStyle = 'red';
  ctx.fill();
}

function drawText(ctx, text, x, y, textColor = 'white', backgroundColor = 'black') {
  ctx.font = '14px Arial';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  
  // Measure text to create background rectangle
  const textMetrics = ctx.measureText(text);
  const textWidth = textMetrics.width;
  const textHeight = 16; // Approximate height for 14px font
  
  // Draw background rectangle
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(x - 2, y - textHeight, textWidth + 4, textHeight + 2);
  
  // Draw text
  ctx.fillStyle = textColor;
  ctx.fillText(text, x, y);
}

function analyzeGaze(annotation, canvasWidth, canvasHeight) {
  const [bx, by, bw, bh] = annotation.bbox;
  const [gx, gy] = annotation.gaze;
  
  // Convert normalized coordinates to pixel coordinates
  const gazeX = gx * canvasWidth;
  const gazeY = gy * canvasHeight;
  const faceX = bx * canvasWidth;
  const faceY = by * canvasHeight;
  const faceW = bw * canvasWidth;
  const faceH = bh * canvasHeight;
  
  // Analyze target type
  let targetType = 'in-frame target';
  if (gazeX < 0 || gazeX > canvasWidth || gazeY < 0 || gazeY > canvasHeight) {
    targetType = 'out-of-frame target';
  } else {
    // Check if gaze is pointing towards camera (eye contact)
    const faceCenterX = faceX + faceW / 2;
    const faceCenterY = faceY + faceH / 2;
    const gazeDistance = Math.sqrt(Math.pow(gazeX - faceCenterX, 2) + Math.pow(gazeY - faceCenterY, 2));
    if (gazeDistance < Math.min(faceW, faceH) * 0.3) {
      targetType = 'Eye-contact';
    }
  }
  
  // Analyze gaze position (distance estimation)
  let gazePosition = 'equal';
  const gazeDistanceFromFace = Math.sqrt(Math.pow(gazeX - (faceX + faceW/2), 2) + Math.pow(gazeY - (faceY + faceH/2), 2));
  const faceSize = Math.sqrt(faceW * faceH);
  
  if (gazeDistanceFromFace > faceSize * 2) {
    gazePosition = 'farther';
  } else if (gazeDistanceFromFace < faceSize * 0.5) {
    gazePosition = 'closer';
  }
  
  // Estimate gaze point distance in meters (rough estimation based on face size)
  // Average human face is about 0.2m wide
  const estimatedFaceWidthMeters = 0.2;
  const pixelsPerMeter = faceW / estimatedFaceWidthMeters;
  const gazeDistanceMeters = (gazeDistanceFromFace / pixelsPerMeter).toFixed(2);
  
  // Simple object detection based on gaze position
  let objectDetection = 'Unknown object';
  if (targetType === 'Eye-contact') {
    objectDetection = 'Camera/Viewer';
  } else if (gazeY < canvasHeight * 0.3) {
    objectDetection = 'Object above (ceiling, sky, etc.)';
  } else if (gazeY > canvasHeight * 0.7) {
    objectDetection = 'Object below (floor, ground, etc.)';
  } else if (gazeX < canvasWidth * 0.3) {
    objectDetection = 'Object on left side';
  } else if (gazeX > canvasWidth * 0.7) {
    objectDetection = 'Object on right side';
  } else {
    objectDetection = 'Central object';
  }
  
  return {
    targetType: targetType,
    gazePosition: gazePosition,
    scaleEstimate: gazeDistanceMeters,
    objectDetection: objectDetection
  };
}

function updateAnnotationsList() {
  const container = document.getElementById('annotations');
  const canvas = document.getElementById('canvas');
  container.innerHTML = '';
  annotations.forEach((ann, idx) => {
    // Perform automatic analysis
    const analysis = analyzeGaze(ann, canvas.width, canvas.height);
    
    const div = document.createElement('div');
    div.className = 'annotation';
    div.innerHTML = `
      <h3>Gaze ${idx + 1}</h3>
      <p>Target type:</p>
      <input type="radio" id="inframe_${idx}" name="target_type_${idx}" value="in-frame target" ${analysis.targetType === 'in-frame target' ? 'checked' : ''}>
      <label for="inframe_${idx}">In-frame target</label><br>
      <input type="radio" id="outframe_${idx}" name="target_type_${idx}" value="out-of-frame target" ${analysis.targetType === 'out-of-frame target' ? 'checked' : ''}>
      <label for="outframe_${idx}">Out-of-frame target</label><br>
      <input type="radio" id="eyecontact_${idx}" name="target_type_${idx}" value="Eye-contact" ${analysis.targetType === 'Eye-contact' ? 'checked' : ''}>
      <label for="eyecontact_${idx}">Eye-contact</label><br>
      <p>Is the gaze target position:</p>
      <input type="radio" id="farther_${idx}" name="farther_closer_${idx}" value="farther" ${analysis.gazePosition === 'farther' ? 'checked' : ''}>
      <label for="farther_${idx}">Farther away</label><br>
      <input type="radio" id="closer_${idx}" name="farther_closer_${idx}" value="closer" ${analysis.gazePosition === 'closer' ? 'checked' : ''}>
      <label for="closer_${idx}">Closer</label><br>
      <input type="radio" id="equal_${idx}" name="farther_closer_${idx}" value="equal" ${analysis.gazePosition === 'equal' ? 'checked' : ''}>
      <label for="equal_${idx}">Equally distant</label><br>
      <input type="radio" id="not_sure_${idx}" name="farther_closer_${idx}" value="not_sure">
      <label for="not_sure_${idx}">Not sure</label><br>
      <p>Estimate the gaze point distance in meters:</p>
      <select id="distance_dropdown_${idx}" onchange="toggleDistanceInput(${idx})" style="width: 100%; margin-top: 5px; padding: 5px;">
        <option value="">Select an option...</option>
        <option value="distance_in_meters">Distance in meters</option>
        <option value="very_far">Very far (not countable or skip the numbers)</option>
      </select>
      <input type="text" id="distance_custom_${idx}" placeholder="Enter distance in meters..." style="width: 100%; margin-top: 5px; padding: 5px; display: none;" value="${analysis.scaleEstimate}"><br>
      <p>Object Detection - Person looking at:</p>
      <select id="object_dropdown_${idx}" onchange="toggleCustomObjectInput(${idx})" style="width: 100%; margin-top: 5px; padding: 5px;">
        <option value="">Select an option...</option>
        <option value="male adult">Male Adult</option>
        <option value="female adult">Female Adult</option>
        <option value="elderly man">Elderly Man</option>
        <option value="elderly woman">Elderly Woman</option>
        <option value="children">Children</option>
        <option value="others">Others</option>
      </select>
      <input type="text" id="object_custom_${idx}" placeholder="Please specify..." style="width: 100%; margin-top: 5px; padding: 5px; display: none;">
      <br><br>
    `;
    container.appendChild(div);
  });
}

function toggleCustomObjectInput(idx) {
  const dropdown = document.getElementById(`object_dropdown_${idx}`);
  const customInput = document.getElementById(`object_custom_${idx}`);
  
  if (dropdown.value === 'others') {
    customInput.style.display = 'block';
    customInput.focus();
  } else {
    customInput.style.display = 'none';
    customInput.value = '';
  }
}

function toggleDistanceInput(idx) {
  const dropdown = document.getElementById(`distance_dropdown_${idx}`);
  const customInput = document.getElementById(`distance_custom_${idx}`);
  
  if (dropdown.value === 'distance_in_meters') {
    customInput.style.display = 'block';
    customInput.focus();
  } else {
    customInput.style.display = 'none';
    if (dropdown.value === 'very_far') {
      customInput.value = '';
    }
  }
}

// VGGT-related functions removed

function resetLastAnnotation() {
  if (annotations.length > 0) {
    // Remove the last annotation
    annotations.pop();
    
    // Reset drawing states
    isDrawingBBox = false;
    isDrawingGaze = false;
    currentAnnotation = null;
    
    // Redraw canvas without the last annotation
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const image = document.getElementById('image');
    redrawCanvas(ctx, image, annotations);
    
    // Update the annotations list display
    updateAnnotationsList();
    
    console.log(`Removed last annotation. Remaining annotations: ${annotations.length}`);
  } else {
    alert('No annotations to reset!');
  }
}

function submitForm() {
  annotations.forEach((ann, idx) => {
    // Add gaze number (1-based indexing to match visual labels)
    ann.gaze_number = idx + 1;
    
    ann.target_type = document.querySelector(`input[name="target_type_${idx}"]:checked`)?.value || '';
    ann.farther_closer = document.querySelector(`input[name="farther_closer_${idx}"]:checked`)?.value || '';
    
    // Handle distance estimation dropdown and custom input
    const distanceDropdown = document.getElementById(`distance_dropdown_${idx}`);
    const distanceInput = document.getElementById(`distance_custom_${idx}`);
    
    if (distanceDropdown.value === 'distance_in_meters' && distanceInput.value.trim() !== '') {
      ann.scale = distanceInput.value.trim();
    } else if (distanceDropdown.value === 'very_far') {
      ann.scale = 'very far (not countable)';
    } else {
      ann.scale = '';
    }
    
    // Handle object detection dropdown and custom input
    const dropdown = document.getElementById(`object_dropdown_${idx}`);
    const customInput = document.getElementById(`object_custom_${idx}`);
    
    if (dropdown.value === 'others' && customInput.value.trim() !== '') {
      ann.object_detection = customInput.value.trim();
    } else if (dropdown.value !== 'others' && dropdown.value !== '') {
      ann.object_detection = dropdown.value;
    } else {
      ann.object_detection = '';
    }
  });
  document.getElementById('annotations_input').value = JSON.stringify(annotations);
  return true;
}
</script>
</head>
<body onload="initCanvas()">
<h2>{{ progress_info }}</h2>
<div class="image-container">
  <img id="image" src="{{ image_url }}">
  <canvas id="canvas" style="position: absolute; top: 0; left: 0;"></canvas>
</div>
<!-- VGGT depth preview UI removed -->
<form method="post" onsubmit="submitForm()">
  <input type="hidden" id="annotations_input" name="annotations">
  <div id="annotations" class="annotations"></div>
  <p>Click and drag to draw a face rectangle, then click to mark gaze target. Repeat for multiple persons.</p>
  <button type="button" onclick="resetLastAnnotation()" style="background-color: #ff6b6b; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px;">Reset Last Annotation</button>
  <input type="submit" value="Submit All" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
</form>
</body>
</html>
    """
    return render_template_string(html, image_url=image_url, progress_info=progress_info, index=index)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)