"""
Dataset Downloader and Organizer for Plant Leaf Disease Detection System.
Collects real leaf images from:
  - PlantVillage (Cercospora, Bacterial Spot, Healthy)
  - Mango Leaf Disease Repository (Anthracnose)
  - ImageNet Sample Repository & OpenCV (Non-Leaf images: faces, animals, vehicles, buildings, fruits, objects)
"""

import os
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DISEASE_DIR = os.path.join(DATASET_DIR, "disease")
VERIFIER_DIR = os.path.join(DATASET_DIR, "leaf_verifier")
TEST_SAMPLES_DIR = os.path.join(BASE_DIR, "test_samples")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))

def download_image(url, save_path):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        # Verify valid image
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path, "JPEG", quality=95)
        return True, save_path
    except Exception as e:
        return False, str(e)

def download_batch_parallel(items, max_workers=16):
    """
    items: list of (url, save_path)
    """
    success = 0
    total = len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, url, path): (url, path) for url, path in items}
        for future in as_completed(futures):
            ok, res = future.result()
            if ok:
                success += 1
    return success, total

def get_plantvillage_file_list(subpath):
    api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{urllib.parse.quote(subpath)}"
    data = fetch_json(api_url)
    return [item["download_url"] for item in data if item["name"].lower().endswith((".jpg", ".jpeg", ".png"))]

def get_mango_anthracnose_file_list():
    api_url = "https://api.github.com/repos/DevyanshMalhotra/MangoLeafDiseaseDetection/contents/dataset/Anthracnose"
    data = fetch_json(api_url)
    return [item["download_url"] for item in data if item["name"].lower().endswith((".jpg", ".jpeg", ".png"))]

def get_imagenet_nonleaf_file_list():
    api_url = "https://api.github.com/repos/EliSchwartz/imagenet-sample-images/contents"
    data = fetch_json(api_url)
    # Non-leaf images (excluding leaves/trees, targeting animals, vehicles, faces, clothes, tools, fruits)
    return [item["download_url"] for item in data if item["name"].endswith(".JPEG")]

def prepare_disease_dataset(target_per_class=300):
    print(f"\n==========================================")
    print(f"1. Fetching URLs for 4 Disease Categories...")
    print(f"==========================================")
    random.seed(42)

    categories = {
        "Cercospora": [],
        "Bacterial_Spot": [],
        "Healthy": [],
        "Anthracnose": []
    }

    # 1. Cercospora
    print("Fetching Cercospora URLs (PlantVillage Corn Cercospora Gray Leaf Spot)...")
    cercospora_urls = get_plantvillage_file_list("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
    random.shuffle(cercospora_urls)
    categories["Cercospora"] = cercospora_urls[:target_per_class]
    print(f"  -> Found {len(cercospora_urls)}, selected {len(categories['Cercospora'])}")

    # 2. Bacterial Spot (from Tomato & Pepper Bacterial Spot)
    print("Fetching Bacterial Spot URLs (PlantVillage Tomato & Pepper Bacterial Spot)...")
    tomato_bact = get_plantvillage_file_list("Tomato___Bacterial_spot")
    pepper_bact = get_plantvillage_file_list("Pepper,_bell___Bacterial_spot")
    bact_urls = tomato_bact + pepper_bact
    random.shuffle(bact_urls)
    categories["Bacterial_Spot"] = bact_urls[:target_per_class]
    print(f"  -> Found {len(bact_urls)}, selected {len(categories['Bacterial_Spot'])}")

    # 3. Healthy (from Tomato Healthy & Corn Healthy)
    print("Fetching Healthy URLs (PlantVillage Tomato & Corn Healthy)...")
    tomato_healthy = get_plantvillage_file_list("Tomato___healthy")
    corn_healthy = get_plantvillage_file_list("Corn_(maize)___healthy")
    healthy_urls = tomato_healthy + corn_healthy
    random.shuffle(healthy_urls)
    categories["Healthy"] = healthy_urls[:target_per_class]
    print(f"  -> Found {len(healthy_urls)}, selected {len(categories['Healthy'])}")

    # 4. Anthracnose (from Mango Leaf Disease dataset)
    print("Fetching Anthracnose URLs (Mango Leaf Anthracnose Dataset)...")
    anthracnose_urls = get_mango_anthracnose_file_list()
    random.shuffle(anthracnose_urls)
    categories["Anthracnose"] = anthracnose_urls[:target_per_class]
    print(f"  -> Found {len(anthracnose_urls)}, selected {len(categories['Anthracnose'])}")

    # Download and organize into train (70%), val (15%), test (15%)
    download_queue = []
    splits = {"train": 0.70, "val": 0.15, "test": 0.15}

    print("\nPreparing train / val / test splits (70% / 15% / 15%)...")
    for cat_name, urls in categories.items():
        n = len(urls)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val

        train_urls = urls[:n_train]
        val_urls = urls[n_train:n_train + n_val]
        test_urls = urls[n_train + n_val:]

        for i, u in enumerate(train_urls):
            dest = os.path.join(DISEASE_DIR, "train", cat_name, f"{cat_name}_train_{i:04d}.jpg")
            download_queue.append((u, dest))
        for i, u in enumerate(val_urls):
            dest = os.path.join(DISEASE_DIR, "val", cat_name, f"{cat_name}_val_{i:04d}.jpg")
            download_queue.append((u, dest))
        for i, u in enumerate(test_urls):
            dest = os.path.join(DISEASE_DIR, "test", cat_name, f"{cat_name}_test_{i:04d}.jpg")
            download_queue.append((u, dest))

    print(f"Starting parallel download of {len(download_queue)} disease images...")
    success, total = download_batch_parallel(download_queue, max_workers=20)
    print(f"Disease images download completed: {success}/{total} successfully downloaded.")

def prepare_leaf_verifier_dataset(target_samples=400):
    print(f"\n==========================================")
    print(f"2. Preparing Leaf-Verifier Dataset...")
    print(f"==========================================")
    random.seed(42)

    # 1. Non-leaf images (ImageNet samples + OpenCV face/fruits/building)
    print("Fetching Non-Leaf URLs (ImageNet + OpenCV samples)...")
    imagenet_urls = get_imagenet_nonleaf_file_list()
    random.shuffle(imagenet_urls)
    
    # Also add known OpenCV sample images (face, animal, fruits, building)
    opencv_samples = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/apple.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg"
    ]
    nonleaf_urls = opencv_samples + imagenet_urls[:target_samples - len(opencv_samples)]
    random.shuffle(nonleaf_urls)

    # Download non-leaf images
    n = len(nonleaf_urls)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    download_queue = []
    for i, u in enumerate(nonleaf_urls[:n_train]):
        dest = os.path.join(VERIFIER_DIR, "train", "non_leaf", f"nonleaf_train_{i:04d}.jpg")
        download_queue.append((u, dest))
    for i, u in enumerate(nonleaf_urls[n_train:n_train + n_val]):
        dest = os.path.join(VERIFIER_DIR, "val", "non_leaf", f"nonleaf_val_{i:04d}.jpg")
        download_queue.append((u, dest))
    for i, u in enumerate(nonleaf_urls[n_train + n_val:target_samples]):
        dest = os.path.join(VERIFIER_DIR, "test", "non_leaf", f"nonleaf_test_{i:04d}.jpg")
        download_queue.append((u, dest))

    print(f"Downloading {len(download_queue)} non-leaf images...")
    s_nonleaf, t_nonleaf = download_batch_parallel(download_queue, max_workers=20)
    print(f"Non-leaf images download completed: {s_nonleaf}/{t_nonleaf}.")

    # 2. Leaf images: Copy or link from already downloaded disease images
    print("Populating leaf images for leaf verifier from verified disease leaf images...")
    disease_train_dir = os.path.join(DISEASE_DIR, "train")
    disease_val_dir = os.path.join(DISEASE_DIR, "val")
    disease_test_dir = os.path.join(DISEASE_DIR, "test")

    def collect_leaf_files(src_dir):
        files = []
        for root, _, filenames in os.walk(src_dir):
            for fn in filenames:
                if fn.endswith((".jpg", ".jpeg", ".png")):
                    files.append(os.path.join(root, fn))
        return files

    leaf_train_pool = collect_leaf_files(disease_train_dir)
    leaf_val_pool = collect_leaf_files(disease_val_dir)
    leaf_test_pool = collect_leaf_files(disease_test_dir)

    random.shuffle(leaf_train_pool)
    random.shuffle(leaf_val_pool)
    random.shuffle(leaf_test_pool)

    def copy_subset(src_files, dst_dir, max_count):
        os.makedirs(dst_dir, exist_ok=True)
        count = 0
        for i, src in enumerate(src_files[:max_count]):
            dst = os.path.join(dst_dir, f"leaf_{i:04d}.jpg")
            try:
                Image.open(src).save(dst, "JPEG", quality=95)
                count += 1
            except Exception:
                pass
        return count

    n_leaf_train = copy_subset(leaf_train_pool, os.path.join(VERIFIER_DIR, "train", "leaf"), n_train)
    n_leaf_val = copy_subset(leaf_val_pool, os.path.join(VERIFIER_DIR, "val", "leaf"), n_val)
    n_leaf_test = copy_subset(leaf_test_pool, os.path.join(VERIFIER_DIR, "test", "leaf"), n - n_train - n_val)
    print(f"Leaf verifier dataset ready: train={n_leaf_train + n_train}, val={n_leaf_val + n_val}, test={n_leaf_test + (n - n_train - n_val)}")

def prepare_curated_test_samples():
    print(f"\n==========================================")
    print(f"3. Preparing Curated Test Samples for Verification...")
    print(f"==========================================")
    os.makedirs(TEST_SAMPLES_DIR, exist_ok=True)

    samples = [
        # Healthy Leaf
        ("https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___healthy/000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL%20Leaf%20259.1.JPG",
         os.path.join(TEST_SAMPLES_DIR, "sample_healthy.jpg")),
        # Cercospora Leaf
        ("https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Corn_(maize)___Cercospora_leaf_spot%20Gray_leaf_spot/00120a18-ff90-46e4-92fb-2b7a10345bd3___RS_GLSp%209357.JPG",
         os.path.join(TEST_SAMPLES_DIR, "sample_cercospora.jpg")),
        # Anthracnose Leaf
        ("https://raw.githubusercontent.com/DevyanshMalhotra/MangoLeafDiseaseDetection/main/dataset/Anthracnose/20211008_124249%20(Custom).jpg",
         os.path.join(TEST_SAMPLES_DIR, "sample_anthracnose.jpg")),
        # Bacterial Spot Leaf
        ("https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Tomato___Bacterial_spot/00416648-be6e-4bd4-bc8d-82f43f8a7240___GCREC_Bact.Sp%203110.JPG",
         os.path.join(TEST_SAMPLES_DIR, "sample_bacterial_spot.jpg")),
        # Non-leaf: Human face (Lena)
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
         os.path.join(TEST_SAMPLES_DIR, "sample_nonleaf_face.jpg")),
        # Non-leaf: Animal (Baboon)
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
         os.path.join(TEST_SAMPLES_DIR, "sample_nonleaf_animal.jpg")),
        # Non-leaf: Fruit (Apple/Fruits)
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
         os.path.join(TEST_SAMPLES_DIR, "sample_nonleaf_fruit.jpg")),
        # Non-leaf: Building
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
         os.path.join(TEST_SAMPLES_DIR, "sample_nonleaf_building.jpg")),
        # Non-leaf: Vehicle (sports car from ImageNet)
        ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n04285008_sports_car.JPEG",
         os.path.join(TEST_SAMPLES_DIR, "sample_nonleaf_car.jpg")),
    ]

    for url, path in samples:
        ok, res = download_image(url, path)
        print(f"  {os.path.basename(path)}: {'OK' if ok else 'FAILED (' + res + ')'}")

if __name__ == "__main__":
    prepare_disease_dataset(target_per_class=250)
    prepare_leaf_verifier_dataset(target_samples=350)
    prepare_curated_test_samples()
    print("\nDataset collection and organization successfully finished!")
