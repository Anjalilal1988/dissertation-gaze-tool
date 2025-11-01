import os
import sys
import zipfile
import tempfile
from urllib.request import urlretrieve


def has_any_files(root):
    try:
        for _root, _dirs, files in os.walk(root):
            if files:
                return True
        return False
    except Exception:
        return False


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {path}: {e}")


def download_and_unpack(url, dest_root):
    print(f"Downloading merged images from {url} ...")
    ensure_dir(dest_root)
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'merged_images.zip')
        try:
            urlretrieve(url, zip_path)
        except Exception as e:
            print(f"Download failed: {e}")
            return False

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest_root)
            print(f"Unpacked archive into {dest_root}")
            return True
        except Exception as e:
            print(f"Unzip failed: {e}")
            return False


def main():
    project_root = os.path.abspath(os.path.dirname(__file__))
    merged_root = os.environ.get('MERGED_ROOT', os.path.join(project_root, 'merged_images'))
    zip_url = os.environ.get('MERGED_ZIP_URL', '').strip()

    print(f"Bootstrap: MERGED_ROOT={merged_root}")
    if has_any_files(merged_root):
        print("Bootstrap: merged images already present; skipping download.")
        return 0

    if not zip_url:
        print("Bootstrap: MERGED_ZIP_URL not set; skipping download. Upload images manually.")
        return 0

    ok = download_and_unpack(zip_url, merged_root)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())