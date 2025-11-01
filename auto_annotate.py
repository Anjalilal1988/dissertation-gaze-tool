import os
import json
import time
from typing import List, Dict, Tuple

try:
    import requests
except ImportError:
    requests = None

from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MERGED_ROOT = os.environ.get('MERGED_ROOT', os.path.join(PROJECT_ROOT, 'merged_images'))
GF_ROOT = os.path.join(MERGED_ROOT, 'gazefollow')
VAT_ROOT = os.path.join(MERGED_ROOT, 'vat')

BASE_URL = os.environ.get('BASE_URL', 'http://127.0.0.1:5000')


def collect_merged_sets() -> Tuple[set, set]:
    gf_rel_paths = set()
    vat_basenames = set()
    try:
        for root, dirs, files in os.walk(GF_ROOT):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), GF_ROOT)
                gf_rel_paths.add(rel.replace(os.sep, '/'))
    except Exception as e:
        print(f"Warning: failed to scan merged GazeFollow: {e}")
    try:
        for root, dirs, files in os.walk(VAT_ROOT):
            for f in files:
                vat_basenames.add(f)
    except Exception as e:
        print(f"Warning: failed to scan merged VAT: {e}")
    return gf_rel_paths, vat_basenames


def load_available_images() -> List[Dict]:
    json_path = os.path.join(PROJECT_ROOT, 'combined_gazefollow_vat.json')
    try:
        with open(json_path, 'r') as f:
            all_images = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        all_images = []
    gf_set, vat_set = collect_merged_sets()
    available = []
    for item in all_images:
        p = item.get('path', '')
        if isinstance(p, str) and (p.startswith('train/') or p.startswith('test2/')):
            if p in gf_set:
                available.append(item)
        else:
            fname = os.path.basename(p)
            if fname in vat_set:
                available.append(item)
    print(f"Available merged images: {len(available)}")
    return available


def find_image_file(base_dir: str, filename: str) -> str:
    base_name = os.path.splitext(filename)[0]
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == filename or os.path.splitext(file)[0] == base_name:
                return os.path.join(root, file)
    return None


def rectify_bbox(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    if w is not None and h is not None:
        if w < 0:
            x = x + w
            w = -w
        if h < 0:
            y = y + h
            h = -h
    return x, y, w, h


def normalize_bbox(bbox: List[float], is_normalized: bool, width: int, height: int) -> List[float]:
    if not bbox or len(bbox) < 4:
        return [0.25, 0.25, 0.5, 0.5]
    x, y, w, h = [float(v) for v in bbox[:4]]
    x, y, w, h = rectify_bbox(x, y, w, h)
    if not is_normalized:
        x /= width
        y /= height
        w /= width
        h /= height
    return [max(0.0, min(1.0, x)), max(0.0, min(1.0, y)), max(0.0, min(1.0, w)), max(0.0, min(1.0, h))]


def normalize_point(pt: List[float], is_normalized: bool, width: int, height: int) -> List[float]:
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
    px_norm = (px / width) if px >= 0 else -0.05
    py_norm = (py / height) if py >= 0 else -0.05
    return [px_norm, py_norm]


def analyze_gaze(bbox: List[float], gaze: List[float], width: int, height: int) -> Dict[str, str]:
    bx, by, bw, bh = bbox
    gx, gy = gaze
    gazeX = gx * width
    gazeY = gy * height
    faceX = bx * width
    faceY = by * height
    faceW = bw * width
    faceH = bh * height
    # Target type
    if gazeX < 0 or gazeX > width or gazeY < 0 or gazeY > height:
        targetType = 'out-of-frame target'
    else:
        faceCenterX = faceX + faceW / 2
        faceCenterY = faceY + faceH / 2
        gazeDistance = ((gazeX - faceCenterX) ** 2 + (gazeY - faceCenterY) ** 2) ** 0.5
        if gazeDistance < min(faceW, faceH) * 0.3:
            targetType = 'Eye-contact'
        else:
            targetType = 'in-frame target'
    # Farther/closer/equal
    gazeDistanceFromFace = ((gazeX - (faceX + faceW/2)) ** 2 + (gazeY - (faceY + faceH/2)) ** 2) ** 0.5
    faceSize = (faceW * faceH) ** 0.5
    if gazeDistanceFromFace > faceSize * 2:
        gazePosition = 'farther'
    elif gazeDistanceFromFace < faceSize * 0.5:
        gazePosition = 'closer'
    else:
        gazePosition = 'equal'
    # Scale estimate in meters
    estimatedFaceWidthMeters = 0.2
    pixelsPerMeter = faceW / estimatedFaceWidthMeters if estimatedFaceWidthMeters > 0 else 1.0
    gazeDistanceMeters = gazeDistanceFromFace / pixelsPerMeter
    scale = f"{gazeDistanceMeters:.2f}"
    # Object detection heuristic
    if targetType == 'Eye-contact':
        objectDetection = 'Camera/Viewer'
    elif gy < 0.3:
        objectDetection = 'Object above (ceiling, sky, etc.)'
    elif gy > 0.7:
        objectDetection = 'Object below (floor, ground, etc.)'
    elif gx < 0.3:
        objectDetection = 'Object on left side'
    elif gx > 0.7:
        objectDetection = 'Object on right side'
    else:
        objectDetection = 'Central object'
    # Focal point (region) classification
    if gx < 0 or gx > 1 or gy < 0 or gy > 1:
        horiz = 'left' if gx < 0 else ('right' if gx > 1 else 'center')
        vert = 'top' if gy < 0 else ('bottom' if gy > 1 else 'middle')
        if horiz == 'center' and vert == 'middle':
            focal = 'out-of-frame'
        elif vert == 'middle':
            focal = f'out-of-frame {horiz}'
        elif horiz == 'center':
            focal = f'out-of-frame {vert}'
        else:
            focal = f'out-of-frame {vert}-{horiz}'
    else:
        horiz = 'left' if gx < 0.3333 else ('right' if gx > 0.6667 else 'center')
        vert = 'top' if gy < 0.3333 else ('bottom' if gy > 0.6667 else 'middle')
        if horiz == 'center' and vert == 'middle':
            focal = 'center'
        elif vert == 'middle':
            focal = horiz
        elif horiz == 'center':
            focal = vert
        else:
            focal = f'{vert}-{horiz}'
    return {
        'target_type': targetType,
        'farther_closer': gazePosition,
        'scale': scale,
        'object_detection': objectDetection,
        'focal_point': focal,
    }


def resolve_image_full_path(item: Dict) -> Tuple[str, int, int]:
    filename = os.path.basename(item['path'])
    if item['path'].startswith('test2/') or item['path'].startswith('train/'):
        full_path = os.path.join(GF_ROOT, item['path'].replace('/', os.sep))
    else:
        full_path = find_image_file(VAT_ROOT, filename) or find_image_file(VAT_ROOT, os.path.splitext(filename)[0] + '.jpg')
    if not full_path or not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found in merged set: {filename}")
    with Image.open(full_path) as img:
        width, height = img.size
    return full_path, width, height


def build_annotations(item: Dict) -> List[Dict]:
    """Build one or more annotations for an image.
    - Primary: dataset-provided gaze (normalized)
    - Secondary (if available and distinct): eye-contact gaze using the eye position
    - Future: if item contains a 'gazes' list, include them all
    """
    full_path, width, height = resolve_image_full_path(item)
    p = item.get('path', '')
    is_gf = isinstance(p, str) and (p.startswith('train/') or p.startswith('test2/'))
    bbox = normalize_bbox(item.get('bbox'), is_gf, width, height)
    eye = normalize_point(item.get('eye'), is_gf, width, height)
    gaze = normalize_point(item.get('gaze') or eye, is_gf, width, height)

    anns: List[Dict] = []

    # Helper to create an annotation dict for a given gaze
    def make_ann(g: List[float]) -> Dict:
        analysis = analyze_gaze(bbox, g, width, height)
        return {
            'bbox': bbox,
            'gaze': g,
            'target_type': analysis['target_type'],
            'farther_closer': analysis['farther_closer'],
            'scale': analysis['scale'],
            'object_detection': analysis['object_detection'],
            'focal_point': analysis['focal_point'],
        }

    # 1) Primary annotation from dataset's gaze
    anns.append(make_ann(gaze))

    # 2) Secondary: eye-contact annotation if distinct enough from primary
    try:
        dx = (gaze[0] - eye[0])
        dy = (gaze[1] - eye[1])
        dist = (dx*dx + dy*dy) ** 0.5
    except Exception:
        dist = 0.0
    if dist > 0.05:  # normalized distance threshold
        anns.append(make_ann(eye))

    # 3) If item contains explicit multiple gazes
    gazes = item.get('gazes')
    if isinstance(gazes, list):
        for g in gazes:
            gp = normalize_point(g, is_gf, width, height)
            anns.append(make_ann(gp))

    # Assign gaze_number sequentially
    for i, ann in enumerate(anns, start=1):
        ann['gaze_number'] = i

    return anns


def main():
    available = load_available_images()
    if not available:
        print("No available merged images. Ensure merged_images is populated and combined_gazefollow_vat.json is valid.")
        return
    if requests is None:
        print("The 'requests' package is required. Please install it: pip install requests")
        return
    sess = requests.Session()
    # Initialize session (sets cookie) and discover user session image count
    r = sess.get(f"{BASE_URL}/")
    if r.status_code not in (200, 302):
        print(f"Failed to initialize session: HTTP {r.status_code}")
        return
    # Load first label page to parse session count
    page = sess.get(f"{BASE_URL}/label_image/0")
    session_total = None
    if page.status_code == 200:
        try:
            import re
            m = re.search(r"Image\s+1\s+of\s+(\d+)", page.text)
            if m:
                session_total = int(m.group(1))
        except Exception:
            session_total = None
    if session_total is None:
        # Fallback: cap at 500 like the app
        session_total = 500
    total = min(len(available), session_total)
    print(f"Auto-annotating {total} images via {BASE_URL}...")
    for idx, item in enumerate(available[:total]):
        anns = build_annotations(item)
        payload = {'annotations': json.dumps(anns)}
        resp = sess.post(f"{BASE_URL}/label_image/{idx}", data=payload)
        if resp.status_code not in (200, 302):
            print(f"[#{idx}] POST failed: HTTP {resp.status_code} - {resp.text[:200]}")
            break
        else:
            print(f"[#{idx+1}/{total}] Saved {len(anns)} annotation(s) for {item.get('path')}")
        time.sleep(0.02)  # gentle pacing
    print("Done. Annotations saved to annotations.json by the app.")


if __name__ == '__main__':
    main()