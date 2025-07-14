import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import time

# Gesture and Morse mappings
GESTURE_TO_MORSE = {0: ".", 1: "-", 2: "x", 3: "submit", 4: "clear", 5: "space", 6: "backspace", 7: "enter", 8: "tab", 9: "shift"}
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', ' ': '/'
}
REVERSE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}

@st.cache_resource
def load_model():
    # For demo purposes, we'll create a mock model
    # Replace this with your actual model loading
    try:
        return tf.keras.models.load_model('gesture_model_augmented.h5')
    except:
        # Mock model for demo
        st.info("Model file not found. Using landmark-based recognition (recommended).")
        return None

model = load_model()

st.title("Hand Gesture Recognition (Webcam) + Morse Code + MediaPipe")
st.write("Show your hand gesture to the webcam and click 'Capture' to add to Morse. Use 'x' gesture to separate letters. Click 'Submit' to decode.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None
        self.hand_roi = None
        self.hand_landmarks = None
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        self.hand_roi = None
        self.hand_landmarks = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.hand_landmarks = hand_landmarks  # Store landmarks for gesture classification
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Calculate bounding box for hand
                h, w, _ = img.shape
                x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add padding
                padding = 30
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Draw bounding box
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Extract hand ROI
                self.hand_roi = img[y_min:y_max, x_min:x_max]
                break
        
        self.frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def preprocess(img):
    """Preprocess image for model input with debug info"""
    try:
        # Convert to grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        # Resize to model expected size
        img_resized = cv2.resize(img_gray, (64, 64))
        
        # Normalize to [0, 1]
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        img_input = np.expand_dims(img_norm, axis=(0, -1))
        
        return img_input
    except Exception as e:
        st.error(f"Preprocessing failed: {str(e)}")
        return None

def classify_gesture_by_landmarks(hand_landmarks):
    """Rule-based gesture classification using hand landmarks"""
    try:
        # Extract key landmarks (normalized coordinates 0-1)
        thumb_tip = hand_landmarks.landmark[4]      # Thumb tip
        thumb_mcp = hand_landmarks.landmark[2]      # Thumb MCP joint
        index_tip = hand_landmarks.landmark[8]      # Index finger tip
        index_pip = hand_landmarks.landmark[6]      # Index finger PIP joint
        middle_tip = hand_landmarks.landmark[12]    # Middle finger tip
        middle_pip = hand_landmarks.landmark[10]    # Middle finger PIP joint
        ring_tip = hand_landmarks.landmark[16]      # Ring finger tip
        ring_pip = hand_landmarks.landmark[14]      # Ring finger PIP joint
        pinky_tip = hand_landmarks.landmark[20]     # Pinky finger tip
        pinky_pip = hand_landmarks.landmark[18]     # Pinky finger PIP joint
        wrist = hand_landmarks.landmark[0]          # Wrist
        
        # Calculate which fingers are extended
        fingers_up = []
        
        # Thumb (check if thumb tip is to the right/left of thumb MCP)
        if thumb_tip.x > thumb_mcp.x:  # Right hand
            fingers_up.append(1)
        else:  # Left hand or thumb down
            fingers_up.append(0)
            
        # Other fingers (check if tip is above PIP joint)
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if tip.y < pip.y:  # Finger is up (y decreases upward)
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Count extended fingers
        fingers_count = sum(fingers_up)
        
        # Gesture classification based on finger patterns
        if fingers_count == 1:
            if fingers_up[1] == 1:  # Only index finger up
                return 0, 0.9  # Dot (.)
            elif fingers_up[0] == 1:  # Only thumb up
                return 1, 0.9  # Dash (-)
            else:
                return 2, 0.8  # Separator (x)
                
        elif fingers_count == 2:
            if fingers_up[1] == 1 and fingers_up[2] == 1:  # Index and middle up
                return 2, 0.9  # Separator (x)
            elif fingers_up[0] == 1 and fingers_up[1] == 1:  # Thumb and index up
                return 1, 0.9  # Dash (-)
            else:
                return 5, 0.8  # Space
                
        elif fingers_count == 3:
            if fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 1:  # Index, middle, ring up
                return 3, 0.9  # Submit
            else:
                return 6, 0.8  # Backspace
                
        elif fingers_count == 4:
            return 4, 0.9  # Clear
            
        elif fingers_count == 5:  # All fingers up
            return 5, 0.9  # Space
            
        elif fingers_count == 0:  # Closed fist
            return 4, 0.9  # Clear
            
        else:
            return 0, 0.5  # Default to dot
            
    except Exception as e:
        st.error(f"Landmark classification failed: {str(e)}")
        return 0, 0.5  # Default to dot

def debug_model_input(img_input, model):
    """Debug function to check model input and output"""
    st.write("**Debug Info:**")
    st.write(f"Input shape: {img_input.shape}")
    st.write(f"Input dtype: {img_input.dtype}")
    st.write(f"Input min/max: {img_input.min():.4f}/{img_input.max():.4f}")
    
    if model is not None:
        prediction = model.predict(img_input, verbose=0)
        st.write(f"Raw prediction shape: {prediction.shape}")
        st.write(f"Raw prediction values: {prediction[0]}")
        st.write(f"Prediction probabilities: {prediction[0] / prediction[0].sum()}")
        
        # Check if all predictions are similar (indicating a problem)
        if np.std(prediction[0]) < 0.1:
            st.warning("⚠️ Model predictions are very similar - this indicates a problem!")
            st.write("Possible issues:")
            st.write("- Model wasn't trained properly")
            st.write("- Input preprocessing doesn't match training")
            st.write("- Model expects different input format")
        
        return prediction
    return None

# Initialize session state
if 'morse_buffer' not in st.session_state:
    st.session_state['morse_buffer'] = ''
if 'current_letter' not in st.session_state:
    st.session_state['current_letter'] = ''
if 'decoded_text' not in st.session_state:
    st.session_state['decoded_text'] = ''
if 'captured_frame' not in st.session_state:
    st.session_state['captured_frame'] = None
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None
if 'last_confidence' not in st.session_state:
    st.session_state['last_confidence'] = None
if 'last_gesture_label' not in st.session_state:
    st.session_state['last_gesture_label'] = None
if 'status_message' not in st.session_state:
    st.session_state['status_message'] = ''

# Create video stream with proper audio disabled
webrtc_ctx = webrtc_streamer(
    key="gesture",
    video_processor_factory=VideoProcessor,
    async_processing=True,
    audio_receiver_size=0,
    media_stream_constraints={"video": True, "audio": False}
)

# Get current frame and hand data
frame = None
hand_roi = None
hand_landmarks = None

if webrtc_ctx.video_processor:
    frame = webrtc_ctx.video_processor.frame
    hand_roi = webrtc_ctx.video_processor.hand_roi
    hand_landmarks = webrtc_ctx.video_processor.hand_landmarks

# Show webcam feed if available
# Show webcam feed and hand detection status side by side
if frame is not None:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image(frame, channels="BGR", caption="Webcam Feed", use_column_width=True)
    with col2:
        if hand_landmarks is not None:
            st.success("✅ Hand detected - Ready to capture gesture! (Using landmark-based recognition)")
        else:
            st.warning("⚠️ No hand detected - Please show your hand to the camera")

else:
    st.info("📷 Webcam is starting... Please allow camera access and wait for video feed.")

# Always show capture controls - moved outside the frame check
st.markdown("### 🎯 Gesture Capture Controls")
col1, col2 = st.columns([3, 1])
with col1:
    capture_button = st.button("🎯 Capture Gesture", type="primary", use_container_width=True)
with col2:
    debug_mode = st.checkbox("Debug", help="Show detailed gesture detection information")

# Always show that we're using landmarks
st.info("🔍 Using landmark-based gesture recognition (recommended)")

# Handle capture button click
if capture_button:
    if frame is None:
        st.error("❌ No webcam feed available. Please ensure camera is working and try again.")
        st.session_state['status_message'] = '❌ No webcam feed - check camera permissions'
    elif hand_landmarks is not None:
        # Always use landmark-based classification
        gesture_id, confidence = classify_gesture_by_landmarks(hand_landmarks)
        st.success("✅ Gesture detected using landmark-based recognition")
        
        if debug_mode:
            st.write("**Landmark-based Classification:**")
            st.write(f"Detected gesture: {gesture_id} ({GESTURE_TO_MORSE.get(gesture_id, 'unknown')})")
            st.write(f"Confidence: {confidence:.2f}")
            
            # Show finger status
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]
            ring_tip = hand_landmarks.landmark[16]
            ring_pip = hand_landmarks.landmark[14]
            pinky_tip = hand_landmarks.landmark[20]
            pinky_pip = hand_landmarks.landmark[18]
            
            # Calculate finger states
            thumb_up = thumb_tip.x > thumb_mcp.x
            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y
            ring_up = ring_tip.y < ring_pip.y
            pinky_up = pinky_tip.y < pinky_pip.y
            
            st.write("**Finger States:**")
            st.write(f"👍 Thumb: {'UP' if thumb_up else 'DOWN'}")
            st.write(f"☝️ Index: {'UP' if index_up else 'DOWN'}")
            st.write(f"🖕 Middle: {'UP' if middle_up else 'DOWN'}")
            st.write(f"💍 Ring: {'UP' if ring_up else 'DOWN'}")
            st.write(f"🤙 Pinky: {'UP' if pinky_up else 'DOWN'}")
            st.write(f"🔢 Fingers up: {sum([thumb_up, index_up, middle_up, ring_up, pinky_up])}")
        
        # Store the frame for display
        if hand_roi is not None and hand_roi.size > 0:
            st.session_state['captured_frame'] = hand_roi.copy()
        else:
            st.session_state['captured_frame'] = frame.copy()
            
        # Store prediction results
        st.session_state['last_prediction'] = gesture_id
        st.session_state['last_confidence'] = confidence
        st.session_state['last_gesture_label'] = GESTURE_TO_MORSE.get(gesture_id, str(gesture_id))
        
        # Process gesture
        gesture_label = GESTURE_TO_MORSE.get(gesture_id, str(gesture_id))
        
        # Update buffers based on gesture
        if gesture_label == 'clear':
            st.session_state['morse_buffer'] = ''
            st.session_state['current_letter'] = ''
            st.session_state['status_message'] = '🗑️ Buffer cleared!'
        elif gesture_label == 'backspace':
            if st.session_state['current_letter']:
                st.session_state['current_letter'] = st.session_state['current_letter'][:-1]
                st.session_state['status_message'] = '⬅️ Removed last symbol from current letter'
            elif st.session_state['morse_buffer']:
                st.session_state['morse_buffer'] = st.session_state['morse_buffer'][:-1]
                st.session_state['status_message'] = '⬅️ Removed last character'
        elif gesture_label == 'space':
            st.session_state['morse_buffer'] += ' '
            st.session_state['current_letter'] = ''
            st.session_state['status_message'] = '␣ Added space'
        elif gesture_label == 'x':
            # Complete current letter
            if st.session_state['current_letter']:
                decoded_char = REVERSE_DICT.get(st.session_state['current_letter'], '?')
                st.session_state['morse_buffer'] += decoded_char
                st.session_state['status_message'] = f"✅ Added letter: {decoded_char} ({st.session_state['current_letter']})"
                st.session_state['current_letter'] = ''
            else:
                st.session_state['status_message'] = '❌ No letter to complete'
        elif gesture_label == 'submit':
            st.session_state['status_message'] = '📝 Use Submit button below to decode'
        elif gesture_label in ['tab', 'enter', 'shift']:
            st.session_state['status_message'] = f'ℹ️ Gesture {gesture_label} ignored'
        elif gesture_label in ['.', '-']:
            st.session_state['current_letter'] += gesture_label
            st.session_state['status_message'] = f'📝 Added {gesture_label} to current letter'
        else:
            st.session_state['status_message'] = f'❓ Unknown gesture: {gesture_label}'
        
        # Force rerun to update UI
        st.rerun()
    else:
        st.error("❌ No hand detected. Please show your hand clearly to the camera.")
        st.session_state['status_message'] = '❌ No hand detected - show your hand to camera'

# Show current status
if st.session_state['status_message']:
    st.info(st.session_state['status_message'])

# Show captured frame and prediction
if st.session_state['captured_frame'] is not None:
    st.image(st.session_state['captured_frame'], channels="BGR", caption="Last Captured Gesture")
    
    if st.session_state['last_prediction'] is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gesture ID", st.session_state['last_prediction'])
        with col2:
            st.metric("Gesture", st.session_state['last_gesture_label'])
        with col3:
            st.metric("Confidence", f"{st.session_state['last_confidence']:.2f}")

# Display current state
st.markdown("### Current State")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Current Letter:** `{st.session_state['current_letter']}`")
with col2:
    st.markdown(f"**Decoded Text:** `{st.session_state['morse_buffer']}`")

# Action buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📝 Submit Final Text", type="secondary"):
        if st.session_state['current_letter']:
            # Auto-complete current letter
            decoded_char = REVERSE_DICT.get(st.session_state['current_letter'], '?')
            st.session_state['morse_buffer'] += decoded_char
            st.session_state['current_letter'] = ''
        st.session_state['decoded_text'] = st.session_state['morse_buffer']
        st.session_state['status_message'] = f"✅ Final text: {st.session_state['decoded_text']}"
        st.rerun()

with col2:
    if st.button("🔄 Complete Current Letter"):
        if st.session_state['current_letter']:
            decoded_char = REVERSE_DICT.get(st.session_state['current_letter'], '?')
            st.session_state['morse_buffer'] += decoded_char
            st.session_state['status_message'] = f"✅ Added letter: {decoded_char} ({st.session_state['current_letter']})"
            st.session_state['current_letter'] = ''
            st.rerun()

with col3:
    if st.button("🗑️ Clear All"):
        st.session_state['morse_buffer'] = ''
        st.session_state['current_letter'] = ''
        st.session_state['decoded_text'] = ''
        st.session_state['status_message'] = '🗑️ Everything cleared!'
        st.rerun()

# Show final result
if st.session_state['decoded_text']:
    st.markdown("### 🎉 Final Result")
    st.markdown(f"**Decoded Text:** `{st.session_state['decoded_text']}`")

# Morse Code Reference and Tools
st.markdown("---")
st.markdown("### 📚 Morse Code Reference & Tools")

# Show gesture mappings
with st.expander("🤲 Gesture Mappings (Landmark-based)"):
    st.markdown("""
    **Landmark-based Gesture Recognition (Always Active):**
    - **☝️ Index finger only**: `.` (dot)
    - **👍 Thumb only**: `-` (dash)  
    - **✌️ Index + Middle**: `x` (complete letter)
    - **👆 Index + Middle + Ring**: `submit` (finalize text)
    - **✊ Closed fist**: `clear` (clear all)
    - **🖐️ All fingers**: `space` (add space)
    - **👍 + ☝️ Thumb + Index**: `-` (dash)
    - **Other combinations**: Various functions
    
    **Tips for better recognition:**
    - Keep your hand steady and well-lit
    - Make clear, distinct gestures
    - Hold the gesture for a moment before capturing
    - Ensure your entire hand is visible in the camera
    """)

# Morse code reference
with st.expander("🔤 Morse Code Reference"):
    st.markdown("**How it works:** Build letters with dots (.) and dashes (-), then complete with 'x' gesture")
    morse_ref = []
    for letter, code in sorted(MORSE_CODE_DICT.items()):
        if letter != ' ':
            morse_ref.append(f"**{letter}**: `{code}`")
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    for i, ref in enumerate(morse_ref):
        if i % 3 == 0:
            col1.markdown(ref)
        elif i % 3 == 1:
            col2.markdown(ref)
        else:
            col3.markdown(ref)

# Text to Morse encoder
st.markdown("### 🔠 Text ↔ Morse Converter")
col1, col2 = st.columns(2)

with col1:
    text_input = st.text_input("Enter Text to Encode")
    if st.button("Encode to Morse"):
        if text_input:
            morse = ' '.join(MORSE_CODE_DICT.get(c.upper(), '?') for c in text_input)
            st.markdown(f"**Morse Code:** `{morse}`")

with col2:
    morse_input = st.text_input("Enter Morse Code to Decode (space-separated)")
    if st.button("Decode from Morse"):
        if morse_input:
            decoded = ''.join(REVERSE_DICT.get(c.strip(), '?') for c in morse_input.split())
            st.markdown(f"**Decoded Text:** `{decoded}`")

# Create sidebar with instructions and troubleshooting
with st.sidebar:
    st.markdown("## 📋 How to Use")
    st.markdown("""
    **Landmark-based Recognition (Always Active):**
    1. **Allow camera access** and wait for the webcam feed to start
    2. **Show your hand** clearly to the camera (green box should appear)
    3. **Make specific gestures**:
       - ☝️ **Index finger only** → dot (.)
       - 👍 **Thumb only** → dash (-)
       - ✌️ **Index + Middle** → complete letter (x)
       - ✊ **Closed fist** → clear all
       - 🖐️ **All fingers** → space
    4. **Click 'Capture Gesture'** to recognize and add to current letter
    5. **Build letters** with dots and dashes, then complete with ✌️ gesture
    6. **Click 'Submit Final Text'** to get your decoded message

    **Example:** To spell "SOS":
    - S: ☝️☝️☝️ (dot dot dot) → ✌️ (complete)
    - O: 👍👍👍 (dash dash dash) → ✌️ (complete)  
    - S: ☝️☝️☝️ (dot dot dot) → ✌️ (complete)

    **Tips:**
    - Keep your hand steady and well-lit
    - Make clear, distinct gestures
    - Wait for the green box to appear before capturing
    - Use debug mode to see finger detection details
    - The system now always uses landmark-based recognition
    """)

    st.markdown("---")
    st.markdown("## 🔧 Troubleshooting")
    st.markdown("""
    - **No hand detected**: Make sure lighting is good and hand is clearly visible
    - **Wrong gesture detected**: Try making the gesture more clearly or use debug mode
    - **Gestures not registering**: Ensure you click 'Capture Gesture' after making the gesture
    - **Camera not starting**: Check browser permissions and refresh the page
    - **Debug mode**: Enable to see detailed finger detection information
    """)

    st.markdown("---")
    st.markdown("## 🎯 Current Setup")
    st.success("✅ Always using landmark-based recognition")
    st.info("📍 More reliable and accurate than model-based detection")