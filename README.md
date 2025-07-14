# Hand Gesture Recognition with Real-time Morse Code Translation

A real-time hand gesture recognition system that uses computer vision and deep learning to detect hand gestures and translate them into Morse code. Built with TensorFlow, OpenCV, and MediaPipe.

## 🎯 Features

- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand detection and tracking
- **Deep Learning Model**: CNN-based gesture recognition with 97.85% accuracy
- **Morse Code Translation**: Converts gestures to Morse code symbols
- **Live Webcam Feed**: Real-time gesture recognition with visual feedback
- **Gesture Mapping**: 10 different gestures mapped to Morse code and special functions

## 📋 Gesture Mapping

| Gesture | Morse Code | Function |
|---------|------------|----------|
| 0 | `.` | Dot |
| 1 | `-` | Dash |
| 2 | `x` | Break |
| 3 | `submit` | Submit/Decode buffer |
| 4 | `clear` | Clear buffer |
| 5 | `space` | Space |
| 6 | `backspace` | Backspace |
| 7 | `enter` | Enter |
| 8 | `tab` | Tab |
| 9 | `shift` | Shift |

---

**Note:** The dataset directory `leapGestRecog/` is not included in this repository and is listed in `.gitignore`. Please download the dataset separately as described below.

## 🖼️ Example Gesture Images

Below are some example gesture images (replace with your own if needed):

| Palm | L | Fist | Fist Moved | Thumb |
|------|---|------|------------|-------|
| ![](examples/gesture_01_palm.png) | ![](examples/gesture_02_l.png) | ![](examples/gesture_03_fist.png) | ![](examples/gesture_04_fist_moved.png) | ![](examples/gesture_05_thumb.png) |

| Index | OK | Palm Moved | C | Down |
|-------|----|------------|---|------|
| ![](examples/gesture_06_index.png) | ![](examples/gesture_07_ok.png) | ![](examples/gesture_08_palm_moved.png) | ![](examples/gesture_09_c.png) | ![](examples/gesture_10_down.png) |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- Good lighting for hand detection

### Installation

1. **Clone or download the project**

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the dataset:**
   ```bash
   kaggle datasets download -d gti-upm/leapgestrecog
   tar -xf leapgestrecog.zip
   ```

### Usage

#### 1. Train the Model (if needed)
```bash
python train_gesture_model.py
```

#### 2. Test the Model
```bash
python test_model.py
```

#### 3. Run Real-time Recognition
```bash
python hand_tracking_gesture.py
```

## 📁 Project Structure

```
Hand Gesture Recognition/
├── train_gesture_model.py          # Training script
├── hand_tracking_gesture.py        # Real-time recognition with hand tracking
├── test_model.py                   # Model testing script
├── requirements.txt                # Python dependencies
├── gesture_model.h5               # Trained model
├── gesture_model_augmented.h5     # Improved model
├── leapGestRecog/                 # Dataset directory
└── README.md                      # This file
```

## 🔧 How It Works

### 1. Hand Detection
- Uses MediaPipe to detect and track hand landmarks
- Creates a bounding box around the detected hand
- Extracts the hand region for processing

### 2. Gesture Recognition
- Preprocesses the hand image (grayscale, resize to 64x64)
- Feeds the image through a trained CNN model
- Predicts the gesture class with confidence score

### 3. Morse Code Translation
- Maps predicted gestures to Morse code symbols
- Builds a buffer of Morse code characters
- Handles special functions (submit, clear, etc.)

## 🎮 Controls

- **Show your hand** in the camera view
- **Make gestures** within the green bounding box
- **Hold gestures steady** for 2-3 seconds
- **Press 'q'** to quit
- **Press 'c'** to clear the Morse buffer

## 📊 Model Performance

- **Test Accuracy**: 97.85%
- **Dataset**: LeapGestRecog (20,000 images)
- **Classes**: 10 different hand gestures
- **Architecture**: CNN with BatchNormalization and Dropout

## 🛠️ Technical Details

### Model Architecture
```
Conv2D(32) → BatchNorm → MaxPool2D
Conv2D(64) → BatchNorm → MaxPool2D
Flatten → Dense(128) → Dropout(0.4) → Dense(10)
```

### Dependencies
- `tensorflow` - Deep learning framework
- `opencv-python` - Computer vision
- `mediapipe` - Hand tracking
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `scikit-learn` - Data processing

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'mediapipe'"**
   ```bash
   pip install mediapipe
   ```

2. **Webcam not detected**
   - Check if webcam is connected
   - Try different camera index (0, 1, 2)

3. **Low accuracy**
   - Ensure good lighting
   - Keep hand steady
   - Position hand in the green box

4. **Model loading error**
   - Check if `gesture_model.h5` exists
   - Re-train the model if needed

### Performance Tips

- **Good lighting**: Ensure your hand is well-lit
- **Clean background**: Avoid cluttered backgrounds
- **Steady hand**: Hold gestures for 2-3 seconds
- **Proper distance**: Keep hand at reasonable distance from camera

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the model architecture
- Adding new gesture mappings

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- MediaPipe for hand tracking
- TensorFlow for deep learning framework 