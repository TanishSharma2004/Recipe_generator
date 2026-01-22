import os

dataset_path = "food_images"

print("=" * 60)
print("Dataset Verification")
print("=" * 60)
print()

if not os.path.exists(dataset_path):
    print(f"✗ Error: '{dataset_path}' folder not found!")
    print("\nMake sure you have the food_images folder in the same directory.")
    exit(1)

print(f"Scanning: {dataset_path}/")
print()

# Count images and classes
total_images = 0
classes = []
class_counts = {}

for root, dirs, files in os.walk(dataset_path):
    # Get class name (subfolder name)
    if root != dataset_path:
        class_name = os.path.basename(root)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            classes.append(class_name)
            count = len(image_files)
            class_counts[class_name] = count
            total_images += count

print(f"Total Classes Found: {len(classes)}")
print(f"Total Images Found: {total_images}")
print()

# Check dataset completeness
if total_images >= 21000:
    print("✓ Full MAFood-121 dataset verified!")
    print("  Status: Ready for training")
elif total_images >= 10000:
    print("⚠ Partial dataset detected")
    print(f"  Expected: ~21,000 images")
    print(f"  Found: {total_images} images")
    print("  You can still train, but with reduced accuracy")
else:
    print("✗ Warning: Dataset seems incomplete")
    print(f"  Found only {total_images} images")
    print("  Recommend: Download full MAFood-121 dataset")

print()
print("=" * 60)

# Show class distribution
if len(classes) > 0:
    print("Sample Classes:")
    for i, (class_name, count) in enumerate(sorted(class_counts.items())[:10]):
        print(f"  {i+1}. {class_name:30s} ({count} images)")
    
    if len(classes) > 10:
        print(f"  ... and {len(classes) - 10} more classes")
    
    print()
    print(f"Average images per class: {total_images // len(classes)}")
else:
    print("✗ No image classes found!")
    print("  Check your folder structure:")
    print("  food_images/")
    print("    ├── Biryani/")
    print("    │   ├── 1_1.jpg")
    print("    │   └── ...")
    print("    ├── Pizza/")
    print("    └── ...")

print("=" * 60)
print()

if total_images > 0:
    print("✓ Dataset verification complete!")
    print("  Next step: python train_model.py")
else:
    print("✗ No images found. Please check your dataset.")

print("=" * 60)