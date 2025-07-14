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
        st.warning("Model file not found. Using mock predictions for demo.")
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
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=(0, -1))
    return img_input

def mock_prediction(img):
    """Mock prediction for demo purposes"""
    # Simulate random gesture prediction
    import random
    gesture_id = random.randint(0, 9)
    confidence = random.uniform(0.6, 0.95)
    return gesture_id, confidence

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

if webrtc_ctx.video_processor:
    frame = webrtc_ctx.video_processor.frame
    hand_roi = webrtc_ctx.video_processor.hand_roi
    
    if frame is not None:
        st.image(frame, channels="BGR", caption="Webcam Feed (with MediaPipe Hand Detection)")
        
        # Capture button - prominently placed
        if st.button("🎯 Capture Gesture", type="primary", use_container_width=True):
            # Use hand ROI if available, else use whole frame
            if hand_roi is not None and hand_roi.size > 0:
                st.session_state['captured_frame'] = hand_roi.copy()
                img_input = preprocess(hand_roi)
            else:
                st.session_state['captured_frame'] = frame.copy()
                img_input = preprocess(frame)
            
            # Make prediction
            if model is not None:
                prediction = model.predict(img_input, verbose=0)
                gesture_id = int(np.argmax(prediction[0]))
                confidence = float(np.max(prediction[0]))
            else:
                gesture_id, confidence = mock_prediction(img_input)
            
            gesture_label = GESTURE_TO_MORSE.get(gesture_id, str(gesture_id))
            
            # Store prediction results
            st.session_state['last_prediction'] = gesture_id
            st.session_state['last_confidence'] = confidence
            st.session_state['last_gesture_label'] = gesture_label
            
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
with st.expander("🤲 Gesture Mappings"):
    st.markdown("""
    - **Gesture 0**: `.` (dot)
    - **Gesture 1**: `-` (dash)
    - **Gesture 2**: `x` (complete letter)
    - **Gesture 3**: `submit` (finalize text)
    - **Gesture 4**: `clear` (clear all)
    - **Gesture 5**: `space` (add space)
    - **Gesture 6**: `backspace` (remove last)
    - **Gesture 7-9**: `enter/tab/shift` (ignored)
    """)

# Morse code reference
with st.expander("🔤 Morse Code Reference"):
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

# Instructions
st.markdown("---")
st.markdown("### 📋 How to Use")
st.markdown("""
1. **Make a gesture** in front of the camera
2. **Click 'Capture Gesture'** to recognize and add to current letter
3. **Use dots (.) and dashes (-)** to build Morse code letters
4. **Use 'x' gesture** or click 'Complete Current Letter' to finish a letter
5. **Click 'Submit Final Text'** to get your decoded message
6. **Use 'space' gesture** to add spaces between words
7. **Use 'clear' gesture** to start over
""")