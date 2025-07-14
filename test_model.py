import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('gesture_model_augmented.h5')

# Path to images
data_dir = "leapGestRecog/leapGestRecog"

# Test on a few sample images from each class
print("Testing model on sample images...")

for class_id in range(10):
    class_dir = os.path.join(data_dir, f"{class_id:02d}")
    if os.path.exists(class_dir):
        # Find first subfolder
        subfolders = [f for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))]
        if subfolders:
            subfolder_path = os.path.join(class_dir, subfolders[0])
            # Find first image
            images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                img_path = os.path.join(subfolder_path, images[0])
                try:
                    # Load and preprocess image
                    img = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=(64, 64))
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    img_input = np.expand_dims(img_array, axis=0)
                    
                    # Make prediction
                    prediction = model.predict(img_input, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    
                    print(f"Class {class_id}: Predicted {predicted_class} with confidence {confidence:.3f}")
                    
                    if predicted_class == class_id:
                        print("  ✓ Correct prediction!")
                    else:
                        print(f"  ✗ Wrong prediction! Expected {class_id}, got {predicted_class}")
                        
                except Exception as e:
                    print(f"Error processing class {class_id}: {e}")

print("\nModel test completed!") 