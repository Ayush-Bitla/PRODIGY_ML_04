import cv2
import numpy as np
import tensorflow as tf
import time
import mediapipe as mp

print("Starting hand tracking gesture recognition...")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load model with error handling
try:
    print("Loading model...")
    model = tf.keras.models.load_model('gesture_model.h5', compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Gesture to Morse code mapping
GESTURE_TO_MORSE = {
    0: ".",
    1: "-", 
    2: "x",  # break
    3: "submit",  # decode buffer
    4: "clear",   # clear buffer
    5: "space",   # space
    6: "backspace", # backspace
    7: "enter",   # enter
    8: "tab",     # tab
    9: "shift"    # shift
}

# Initialize webcam
print("Initializing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

print("Webcam initialized successfully!")

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables
morse_buffer = ""
last_gesture = None
frame_count = 0
prediction_threshold = 5
last_prediction_time = time.time()
prediction_cooldown = 2.0  # 2 seconds between predictions

print("Hand Tracking Gesture Recognition Started!")
print("Press 'q' to quit, 'c' to clear morse buffer")
print("Show your hand to the camera")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        hand_detected = False
        hand_roi = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get hand bounding box
                h, w, _ = frame.shape
                x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add padding to the bounding box
                padding = 50
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Extract hand ROI
                hand_roi = frame[y_min:y_max, x_min:x_max]
                hand_detected = True
                
                # Add hand detection status
                cv2.putText(display_frame, "HAND DETECTED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                break  # Only process the first hand
        
        if not hand_detected:
            cv2.putText(display_frame, "NO HAND DETECTED", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Process hand ROI if detected
        if hand_detected and hand_roi is not None and hand_roi.size > 0:
            try:
                # Preprocess hand ROI for model
                roi_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (64, 64))
                roi_normalized = roi_resized / 255.0
                roi_input = np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)
                
                # Make prediction
                prediction = model.predict(roi_input, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                current_time = time.time()
                
                # Only process if confidence is high enough and enough time has passed
                if confidence > 0.8 and (current_time - last_prediction_time) > prediction_cooldown:
                    if predicted_class == last_gesture:
                        frame_count += 1
                    else:
                        frame_count = 1
                        last_gesture = predicted_class
                    
                    # If we have consistent predictions for threshold frames
                    if frame_count >= prediction_threshold:
                        last_prediction_time = current_time
                        
                        # Add to morse buffer based on gesture
                        if predicted_class in GESTURE_TO_MORSE:
                            morse_symbol = GESTURE_TO_MORSE[predicted_class]
                            
                            # Handle special gestures
                            if morse_symbol == "submit":
                                print(f"Morse Code: {morse_buffer}")
                                morse_buffer = ""
                            elif morse_symbol == "clear":
                                morse_buffer = ""
                                print("Buffer cleared")
                            elif morse_symbol == "space":
                                morse_buffer += " "
                            elif morse_symbol == "backspace":
                                morse_buffer = morse_buffer[:-1] if morse_buffer else ""
                            elif morse_symbol == "enter":
                                morse_buffer += "\n"
                            elif morse_symbol == "tab":
                                morse_buffer += "\t"
                            elif morse_symbol == "shift":
                                pass
                            else:
                                # Add regular morse symbols
                                morse_buffer += morse_symbol
                            
                            print(f"Gesture: {predicted_class} ({morse_symbol}) - Confidence: {confidence:.2f}")
                
                # Display information on frame
                cv2.putText(display_frame, f"Gesture: {predicted_class}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Morse Buffer: {morse_buffer}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(display_frame, f"Frame Count: {frame_count}/{prediction_threshold}", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
            except Exception as e:
                print(f"Error processing hand ROI: {e}")
                continue
        
        # Display instructions
        cv2.putText(display_frame, "Press 'q' to quit, 'c' to clear buffer", (10, display_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Hand Tracking Gesture Recognition', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            morse_buffer = ""
            print("Buffer cleared")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Clean up
    print("Cleaning up...")
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Hand tracking gesture recognition stopped.") 