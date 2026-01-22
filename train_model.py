import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # Faster than EfficientNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# Configuration - OPTIMIZED FOR SPEED
IMG_SIZE = 128  # Reduced from 224 (4x faster!)
BATCH_SIZE = 64  # Larger batches = fewer iterations
EPOCHS = 25  # Reduced from 50
DATASET_PATH = "food_images"

print("="*60)
print("FAST TRAINING MODE - Optimized for PC")
print("="*60)
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE} (faster processing)")
print(f"Batch Size: {BATCH_SIZE} (larger batches)")
print(f"Epochs: {EPOCHS} (reduced training time)")
print(f"Expected Time: 1-3 hours on decent PC")
print("="*60)

# Lighter data augmentation for speed
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Reduced
    width_shift_range=0.1,  # Reduced
    height_shift_range=0.1,  # Reduced
    zoom_range=0.1,  # Reduced
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load training data
print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
print("Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get number of classes
num_classes = len(train_generator.class_indices)
print(f"\nNumber of classes: {num_classes}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Save class indices
import json
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)
print("Class labels saved to 'class_labels.json'")

# Build FASTER model using MobileNetV2 (lighter than EfficientNet)
print("\nBuilding model with MobileNetV2 (optimized for speed)...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model
base_model.trainable = False

# Simpler architecture for speed
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # Reduced from 512
x = Dropout(0.3)(x)  # Reduced dropout
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile with higher learning rate for faster convergence
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

print("\nModel compiled successfully!")
print(f"Total parameters: {model.count_params():,}")

# Aggressive callbacks for faster training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,  # Reduced from 7
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,  # Reduced from 3
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n" + "="*60)
print("PHASE 1: Transfer Learning (10 epochs)")
print("="*60)
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# Fine-tuning with fewer layers
print("\n" + "="*60)
print("PHASE 2: Fine-tuning (15 epochs)")
print("="*60)
base_model.trainable = True
# Freeze all but last 10 layers (faster than 20)
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=10,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save('food_recognition_model.keras')
print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print("Model saved as 'food_recognition_model.keras'")

# Final evaluation
print("\nEvaluating model...")
val_loss, val_acc, val_top5 = model.evaluate(val_generator, verbose=0)
print(f"\nFinal Results:")
print(f"  Validation Accuracy: {val_acc*100:.2f}%")
print(f"  Top-5 Accuracy: {val_top5*100:.2f}%")
print(f"  Validation Loss: {val_loss:.4f}")

print("\n" + "="*60)
print("Next Steps:")
print("  1. Test: python test_prediction.py food_images/Biryani/1_2.jpg")
print("  2. Run app: streamlit run app.py")
print("="*60)