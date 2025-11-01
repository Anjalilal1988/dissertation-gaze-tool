# Gaze Detection Annotation Tool

Lightweight Flask app to label gaze annotations on images. Draw a face bounding box, click the gaze target, and submit — annotations are saved into `annotations.json` with automatic indexing.

## Features
- Label multiple persons per image with gaze lines and numbered labels.
- Simple form fields: target type, distance estimate, and object category.
- Serves a fixed set of up to 500 images from `merged_images` so all users see the same dataset.
- Image paths resolved from `merged_images/gazefollow` and `merged_images/vat`.

## Requirements
- Python 3.9+
- Pip (`pip --version`)

## Setup
- (Optional) create a virtual environment:
  - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
  - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Ensure data exists:
  - `combined_gazefollow_vat.json` in repo root.
  - Images copied to `merged_images/gazefollow/...` and `merged_images/vat/...`.

## Run
- Default port: `python app.py`
- Custom port (Windows): `$env:PORT=5001; python app.py`
- Custom port (macOS/Linux): `PORT=5001 python app.py`
- Open: `http://127.0.0.1:5000/` (or your chosen port)

### One-click (Windows)
- Double-click `run.bat`.
- Optional: pass a port as the first argument, e.g., `run.bat 5001`.

## Usage
- The app redirects to the first image and renders the labeling UI.
- Draw rectangle → click gaze point → repeat as needed.
- Use “Reset Last Annotation” to undo the most recent entry.
- Click “Submit All” to save for the current image and advance to the next.
- Saved to `annotations.json` as:
  - `[{ "index": <auto_id>, "annotations": [{ bbox, gaze, gaze_number, ... }] }, ... ]`

## Data Notes
- This repository ignores heavy datasets by default via `.gitignore`:
  - `Gazefollow/`, `VAT/images/`, `merged_images/`, and `labels.csv` are excluded.
  - If you need a few demo images in Git, place them under `samples/` and adjust `.gitignore` accordingly.

## Participant Kit (Local-and-send)
- Prepare a zip for participants that includes:
  - This project folder and `merged_images/gazefollow/...` and `merged_images/vat/...` with the fixed 500 images.
  - `run.bat` (Windows) or instructions to run `python app.py`.
- Participant run (Windows):
  - Optional: set a personalized annotations path (replaces default `annotations.json`):
    - `set ANNOTATIONS_PATH=%CD%\annotations_<ID>.json` then `python app.py`
  - Or run `run.bat` and rename `annotations.json` to `annotations_<ID>.json` before sending.
- Participant run (macOS/Linux):
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - Optional personalized path: `ANNOTATIONS_PATH=$PWD/annotations_<ID>.json python3 app.py`
  - Otherwise run `python3 app.py` and rename the output file.

### Collecting and Merging Participant Files
- Ask each participant to send their `annotations_<ID>.json` back.
- Put all files into `participants/` and run:
  - `python merge_annotations.py participants/*.json annotations_merged.json --tag-annotator`
- Result: `annotations_merged.json` groups annotations by `image_path`, assigns new sequential indices, and tags each entry with `annotator_id`.
- Keep the original `annotations.json` files for audit; `annotations_merged.json` is your master file for analysis.

## GitHub
- Suggested repo: `https://github.com/anjalisehgal1988-arch/gaze_detection`
- Initial push:
  - `git init && git branch -M main`
  - `git add . && git commit -m "Initial commit"`
  - `git remote add origin https://github.com/anjalisehgal1988-arch/gaze_detection.git`
  - `git push -u origin main`

### If origin already points to Thesis
- Update remote URL:
  - `git remote set-url origin https://github.com/anjalisehgal1988-arch/gaze_detection.git`

## Environment Variables
- `PORT`: HTTP port to bind (default `5000`).
- `ANNOTATIONS_PATH`: where to save annotations (default `./annotations.json`).
  - In production, set to a persistent disk path, e.g., `/var/data/annotations.json`.
- `MERGED_ROOT`: base folder for images (default `./merged_images`).
  - In production, point to a mounted volume, e.g., `/var/data/merged_images`.

## Deployment (Render)
- This repository includes `render.yaml` for one-click deployment.
- Steps:
  - Push to GitHub.
  - On Render: New → Blueprint → select this repo.
  - Render will provision a Web Service with:
    - Build: `pip install -r requirements.txt`
    - Start: `python bootstrap_images.py && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1`
    - Disk mounted at `/var/data`.
    - Env vars:
      - `ANNOTATIONS_PATH=/var/data/annotations.json`
      - `MERGED_ROOT=/var/data/merged_images`
      - (Optional) `MERGED_ZIP_URL=https://.../merged_images.zip` — if set, the service will download and unpack images into `/var/data/merged_images` on first start.
  - Upload your images to the mounted disk (`/var/data/merged_images`) or provide a zip via `MERGED_ZIP_URL`.
  - Open the public URL; annotations persist to `/var/data/annotations.json`.

### Multi-user notes
- To avoid concurrent write issues while multiple users annotate, the service runs with a single Gunicorn worker.
- Annotations are saved under a file lock with `portalocker`, and each image gets a reserved annotation index for uniqueness.
- Each saved annotation now includes `image_path` so you can link labels to source images reliably.

## License
- See `LICENSE` for terms.