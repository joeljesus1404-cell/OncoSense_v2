import shutil, os, glob

os.makedirs('data/breakhis/benign', exist_ok=True)
os.makedirs('data/breakhis/malignant', exist_ok=True)

base = 'data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'

for img in glob.glob(f'{base}/benign/**/*.png', recursive=True):
    shutil.copy2(img, 'data/breakhis/benign/')

for img in glob.glob(f'{base}/malignant/**/*.png', recursive=True):
    shutil.copy2(img, 'data/breakhis/malignant/')

print(f"Benign: {len(os.listdir('data/breakhis/benign'))} images")
print(f"Malignant: {len(os.listdir('data/breakhis/malignant'))} images")