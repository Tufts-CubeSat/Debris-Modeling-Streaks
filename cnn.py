import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import cv2

def load_debris_dataset(dataset_path='training_dataset', img_size=(256, 256)):
    """
    Load the debris streak dataset from directory structure.

    Args:
        dataset_path: Path to dataset directory containing 'streaks' and 'non-streaks' folders
        img_size: Target size to resize images to (height, width)

    Returns:
        x_data: Array of images
        y_data: Array of labels (0=non-streak, 1=streak)
    """
    images = []
    labels = []

    # Load streaks
    
    streak_dir = Path(dataset_path) / 'streaks' / 'images'
    print(f"Loading streak images from {streak_dir}...")
    for img_path in sorted(streak_dir.glob('*.png')):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize to target size
            img = cv2.resize(img, img_size)
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            # Add channel dimension
            img = np.expand_dims(img, axis=-1)
            images.append(img)
            labels.append(1)

    print(f"Loaded {len([l for l in labels if l == 1])} streak images")

    # Load non-streaks
    non_streak_dir = Path(dataset_path) / 'non-streaks' / 'images'
    print(f"Loading non-streak images from {non_streak_dir}...")
    for img_path in sorted(non_streak_dir.glob('*.png')):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize to target size
            img = cv2.resize(img, img_size)
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            # Add channel dimension
            img = np.expand_dims(img, axis=-1)
            images.append(img)
            labels.append(0)

    print(f"Loaded {len([l for l in labels if l == 0])} non-streak images")

    # Convert to numpy arrays
    x_data = np.array(images)
    y_data = np.array(labels)

    # Shuffle the dataset
    indices = np.random.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]

    print(f"\nDataset shape: {x_data.shape}")
    print(f"Labels shape: {y_data.shape}")
    print(f"Class distribution: {np.bincount(y_data)}")

    return x_data, y_data

# Load the training dataset
print("="*60)
print("Loading training dataset...")
print("="*60)
x_train, y_train = load_debris_dataset(dataset_path='train_dataset', img_size=(256, 256))

# Load the test dataset
print("\n" + "="*60)
print("Loading test dataset...")
print("="*60)
x_test, y_test = load_debris_dataset(dataset_path='test_dataset_new', img_size=(256, 256))

print("\n" + "="*60)
print(f"Training set: {x_train.shape[0]} images")
print(f"Test set: {x_test.shape[0]} images")
print("="*60)

# Build the CNN model for binary classification
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(256, 256, 1)),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Setup callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(
    monitor='val_loss',           # Monitor validation loss
    patience=5,                   # Stop if no improvement for 5 epochs
    restore_best_weights=True,    # Restore weights from best epoch
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_streak_classifier.keras',  # Save to this file
    monitor='val_loss',            # Monitor validation loss
    save_best_only=True,           # Only save when val_loss improves
    mode='min',                    # Lower val_loss is better
    verbose=1
)

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"\nTest accuracy = {test_acc:.3f}")
print(f"Test loss = {test_loss:.3f}")

# Save final model (best weights already saved by callback)
model.save('final_streak_classifier.keras')
print("\nFinal model saved as 'final_streak_classifier.keras'")
print("Best model (lowest val_loss) saved as 'best_streak_classifier.keras'")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='train')
ax1.plot(history.history['val_accuracy'], label='val')
ax1.legend()
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')

ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='val')
ax2.legend()
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history saved as 'training_history.png'")
plt.show()