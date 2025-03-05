import os
import shutil
import random

# Define paths to your directories
base_dir = 'DATA/01_lesions'  # Base directory
train_dir = f'{base_dir}/train'
val_dir = f'{base_dir}/val'

# Percentage of data to use for validation
val_split = 0.2  # 20% for validation

# Create validation directory if it doesn't exist
os.makedirs(val_dir, exist_ok=True)

# Get only .jpg files (ignore .txt or other non-image files)
files = [f for f in os.listdir(train_dir) if f.lower().endswith('.jpg')]
random.shuffle(files)

# Determine the number of validation samples
val_size = int(len(files) * val_split)

# Move files to the validation directory
for file_name in files[:val_size]:
    src_path = f'{train_dir}/{file_name}'
    dest_path = f'{val_dir}/{file_name}'
    shutil.move(src_path, dest_path)

print(f"Successfully moved {val_size} images to the validation set.")
