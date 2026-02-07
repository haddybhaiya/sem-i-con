import os
import shutil
import random
import glob

# --- Configuration ---
SOURCE_DIR_1 = "dataset/1test"
SOURCE_DIR_2 = "dataset/test" # Assuming this is your "fuzzy" test data location
DEST_DIR = "dataset/2test"

IMAGES_PER_CLASS_PER_SOURCE = 25
TOTAL_IMAGES_PER_CLASS = IMAGES_PER_CLASS_PER_SOURCE * 2

# Define the 8 classes (Alphabetical order is safest)
CLASSES = [
    "bridge", "clean", "cmp", "crack", 
    "ler", "open", "other", "via"
]

def create_balanced_test_set():
    # 1. Create the destination root directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created destination directory: {DEST_DIR}")

    for class_name in CLASSES:
        # Define source paths for this specific class
        source1_path = os.path.join(SOURCE_DIR_1, class_name)
        source2_path = os.path.join(SOURCE_DIR_2, class_name)
        
        # Define destination path for this specific class
        dest_class_path = os.path.join(DEST_DIR, class_name)
        if not os.path.exists(dest_class_path):
            os.makedirs(dest_class_path)

        # 2. Collect image paths from both sources
        images1 = glob.glob(os.path.join(source1_path, "*.*"))
        images2 = glob.glob(os.path.join(source2_path, "*.*"))

        # 3. Randomly sample N images from each source
        if len(images1) >= IMAGES_PER_CLASS_PER_SOURCE:
            sample1 = random.sample(images1, IMAGES_PER_CLASS_PER_SOURCE)
        else:
            print(f"Warning: Not enough images in {source1_path}. Taking all {len(images1)}.")
            sample1 = images1
            
        if len(images2) >= IMAGES_PER_CLASS_PER_SOURCE:
            sample2 = random.sample(images2, IMAGES_PER_CLASS_PER_SOURCE)
        else:
            print(f"Warning: Not enough images in {source2_path}. Taking all {len(images2)}.")
            sample2 = images2

        # 4. Copy the selected images to the destination folder
        all_samples = sample1 + sample2
        for img_path in all_samples:
            # Generate a unique destination name to prevent overwrites
            # using the original filename and a random component
            filename = os.path.basename(img_path)
            # Use shutil.copy to duplicate the file
            shutil.copy(img_path, os.path.join(dest_class_path, filename))

        print(f"✅ Class '{class_name}': Copied {len(all_samples)} images to {dest_class_path}")

    print("\n🎉 Combined dataset '2test' creation complete!")

if __name__ == "__main__":
    create_balanced_test_set()
