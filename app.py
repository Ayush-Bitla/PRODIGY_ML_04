import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp

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
    return tf.keras.models.load_model('gesture_model_augmented.h5')

model = load_model()

st.title("Hand Gesture Recognition (Webcam) + Morse Code + MediaPipe")
st.write("Show your hand gesture to the webcam and click 'Capture' to add to Morse. Click 'Submit' to decode. Now using MediaPipe for hand detection!")

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
                h, w, _ = img.shape
                x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                padding = 30
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
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

# Session state for Morse buffer and decoded text
if 'morse_buffer' not in st.session_state:
    st.session_state['morse_buffer'] = ''
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

webrtc_ctx = webrtc_streamer(key="gesture", video_processor_factory=VideoProcessor, async_processing=True, audio_receiver_size=0, media_stream_constraints={"video": True, "audio": False})

if webrtc_ctx.video_processor:
    frame = webrtc_ctx.video_processor.frame
    hand_roi = webrtc_ctx.video_processor.hand_roi
    if frame is not None:
        st.image(frame, channels="BGR", caption="Webcam Feed (with MediaPipe)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Capture"):
                # Use hand ROI if available, else use whole frame
                if hand_roi is not None and hand_roi.size > 0:
                    st.session_state['captured_frame'] = hand_roi.copy()
                    img_input = preprocess(hand_roi)
                else:
                    st.session_state['captured_frame'] = frame.copy()
                    img_input = preprocess(frame)
                prediction = model.predict(img_input, verbose=0)
                gesture_id = int(np.argmax(prediction[0]))
                confidence = float(np.max(prediction[0]))
                gesture_label = GESTURE_TO_MORSE.get(gesture_id, str(gesture_id))
                st.session_state['last_prediction'] = gesture_id
                st.session_state['last_confidence'] = confidence
                st.session_state['last_gesture_label'] = gesture_label
                # Update Morse buffer
                if gesture_label == 'clear':
                    st.session_state['morse_buffer'] = ''
                elif gesture_label == 'backspace':
                    st.session_state['morse_buffer'] = st.session_state['morse_buffer'][:-1]
                elif gesture_label == 'space':
                    st.session_state['morse_buffer'] += ' '
                elif gesture_label == 'submit':
                    pass  # Use Submit button to decode
                elif gesture_label in ['tab', 'enter', 'shift']:
                    pass  # Ignore for Morse buffer
                else:
                    st.session_state['morse_buffer'] += gesture_label
        with col2:
            if st.button("Submit"):
                # Decode Morse buffer to text
                words = st.session_state['morse_buffer'].strip().split('x')
                decoded = ''.join([REVERSE_DICT.get(w, '?') for w in words])
                st.session_state['decoded_text'] = decoded
                st.session_state['morse_buffer'] = ''

if st.session_state['captured_frame'] is not None:
    st.image(st.session_state['captured_frame'], channels="BGR", caption="Captured Hand ROI" if st.session_state['captured_frame'].shape[0] < 400 else "Captured Frame")
    if st.session_state['last_prediction'] is not None:
        st.markdown(f"**Prediction:** Gesture {st.session_state['last_prediction']} ({st.session_state['last_gesture_label']})")
        st.markdown(f"**Confidence:** {st.session_state['last_confidence']:.2f}")

st.markdown(f"### Morse Buffer: `{st.session_state['morse_buffer']}`")
st.markdown(f"### Decoded Text: `{st.session_state['decoded_text']}`")

if st.button("Clear Buffer"):
    st.session_state['morse_buffer'] = ''
    st.session_state['decoded_text'] = ''

# Encoder/Decoder UI
st.markdown("---")
st.markdown("### 🔠 Morse Code ↔ Text")
col3, col4 = st.columns(2)
with col3:
    text_input = st.text_input("Enter Text to Encode")
    if st.button("Encode"):
        morse = ' '.join(MORSE_CODE_DICT.get(c.upper(), '') for c in text_input)
        st.markdown(f"**Morse Code:** `{morse}`")
with col4:
    morse_input = st.text_input("Enter Morse Code to Decode (use space between letters)")
    if st.button("Decode"):
        decoded = ''.join(REVERSE_DICT.get(c, '?') for c in morse_input.strip().split())
        st.markdown(f"**Decoded Text:** `{decoded}`") 