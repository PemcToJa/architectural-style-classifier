import os
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm

DATASET_DIR = r"C:\Users\przem\Desktop\architectural-style-classifier\data\raw\Architecture_art_styles"
PROJECT_DIR = r"C:\Users\przem\Desktop\architectural-style-classifier"
DUPLICATES_DIR = os.path.join(PROJECT_DIR, "data", "duplicates_found")
HASH_SIZE = 16

os.makedirs(DUPLICATES_DIR, exist_ok=True)

hash_dict = {}
duplicates = []

print("\n================================")
print("Rozpoczynam skanowanie datasetu")
print("================================")

subfolders = sorted([f.path for f in os.scandir(DATASET_DIR) if f.is_dir()])

for folder in subfolders:
    folder_name = os.path.basename(folder)
    print(f"\nAktualnie skanuję folder: {folder_name}")

    file_list = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

    for fname in tqdm(file_list, desc=f"  → Pliki w {folder_name}", leave=False):
        path = os.path.join(folder, fname)

        try:
            img = Image.open(path).convert("RGB")
            img_hash = str(imagehash.phash(img, hash_size=HASH_SIZE))
        except Exception as e:
            print(f"Błąd odczytu: {path} — {e}")
            continue

        if img_hash in hash_dict:
            original = hash_dict[img_hash]
            duplicate = path
            duplicates.append((original, duplicate))

            pair_dir = os.path.join(DUPLICATES_DIR, f"pair_{len(duplicates)}")
            os.makedirs(pair_dir, exist_ok=True)

            shutil.copy2(original, os.path.join(pair_dir, "original.jpg"))
            shutil.copy2(duplicate, os.path.join(pair_dir, "duplicate.jpg"))

        else:
            hash_dict[img_hash] = path

print("\n========================================================================================================")
print(f"Znaleziono {len(duplicates)} duplikatów.")
print(f"Folder z parami duplikatów: {DUPLICATES_DIR}")
print("========================================================================================================\n")

if duplicates:
    print("Wybierz opcję:")
    print("1 - usuń wszystkie duplikaty automatycznie")
    print("0 - zostaw wszystkie duplikaty bez zmian")

    choice = input("Twój wybór (1/0): ").strip()

    if choice == "1":
        for _, duplicate in duplicates:
            os.remove(duplicate)
        print(f"Usunięto wszystkie duplikaty ({len(duplicates)})")

    if choice == "0":
        print("Zostawiono wszystkie duplikaty.")

input("\nNaciśnij Enter, aby usunąć cały folder duplicates_found wraz z zawartością...")
shutil.rmtree(DUPLICATES_DIR)
print("Folder duplicates_found został usunięty.")

