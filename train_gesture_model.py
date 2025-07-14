import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# Path to images (update if needed)
data_dir = "leapGestRecog/leapGestRecog"

# We expect subfolders for each gesture (0-9)
classes = sorted(os.listdir(data_dir))
print("Classes found:", classes)

images = []
labels = []

IMG_SIZE = 64

def load_images():
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        print(f"Processing class {label}...")
        
        # Check if the class directory exists
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping...")
            continue
            
        for subfolder in os.listdir(class_dir):
            subfolder_path = os.path.join(class_dir, subfolder)
            
            # Skip if not a directory
            if not os.path.isdir(subfolder_path):
                continue
                
            print(f"  Processing subfolder: {subfolder}")
            
            for img_file in os.listdir(subfolder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subfolder_path, img_file)
                    try:
                        img = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
                        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(int(label))
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue

print("Loading images...")
load_images()
print(f"Loaded {len(images)} images.")

X = np.array(images)
y = to_categorical(np.array(labels), num_classes=len(classes))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Build simple CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model and training stats
model.save("gesture_model.h5")

# Plot accuracy and loss
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.savefig('training_plots.png')
plt.show()

# Save history stats to a file
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

print("Model and training history saved!") 