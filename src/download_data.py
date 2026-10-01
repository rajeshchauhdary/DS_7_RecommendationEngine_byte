import io
import os
import ssl
import urllib.request
import zipfile

DATA_DIR = "data"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_and_extract_movielens(url=MOVIELENS_URL, target_dir=DATA_DIR):
    os.makedirs(target_dir, exist_ok=True)
    extract_check_file = os.path.join(target_dir, "ml-100k", "u.data")

    if os.path.exists(extract_check_file):
        print("Dataset already downloaded and extracted.")
        return

    print("Fetching MovieLens 100k dataset archive...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, context=ctx) as response:
        archive_bytes = response.read()

    print("Extracting dataset...")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zip_ref:
        zip_ref.extractall(target_dir)

    print(f"Dataset successfully extracted to {target_dir}/ml-100k")

if __name__ == "__main__":
    download_and_extract_movielens()
