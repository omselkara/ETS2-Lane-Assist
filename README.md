# ETS2 Lane Assist 🚛🤖

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Platform-Windows-lightgrey.svg" alt="Platform">
</p>

**ETS2 Lane Assist** is an AI-powered lane keeping assistance system for Euro Truck Simulator 2 (ETS2). Using deep learning and computer vision, this project enables autonomous steering control to keep your truck in the center of the lane.

## 🎯 Features

- **🧠 Deep Learning Powered**: Uses a Convolutional Neural Network (CNN) to predict steering angles based on road images
- **🎮 Real-time Control**: Processes game screen in real-time and provides immediate steering corrections
- **📊 Telemetry Integration**: Reads game data through ETS2 SDK telemetry for accurate vehicle control
- **🚗 Cruise Control Integration**: Automatically maintains speed at 80 km/h using cruise control
- **📸 Dataset Collection**: Built-in tools to collect your own driving data for model training
- **🔧 Customizable**: Train your own model with your driving style
- **⚡ Edge Detection**: Advanced edge detection algorithm for better lane recognition

## 📋 Requirements

### Software Dependencies
- **Operating System**: Windows (required for DirectX and game integration)
- **Python**: 3.7 or higher
- **Euro Truck Simulator 2**: Game must be installed

### Python Packages
```
tensorflow>=2.0
keras
numpy
opencv-python (cv2)
pillow
pyautogui
pynput
keyboard
win32gui (pywin32)
win32api (pywin32)
tqdm
colorama
```

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/omselkara/ETS2-Lane-Assist.git
cd ETS2-Lane-Assist
```

### 2. Install Python Dependencies
```bash
pip install tensorflow keras numpy opencv-python pillow pyautogui pynput keyboard pywin32 tqdm colorama
```

### 3. Extract the Model Files
- Extract the files from `model.part1.rar` and `model.part2.rar` (multi-part archive)
- Place the extracted `model.keras` file in the same directory as `play.py`

### 4. Install ETS2 Telemetry Plugin
- Copy the `ets2-telemetry.dll` file
- Paste it into your Euro Truck Simulator 2 installation folder:
  ```
  [ETS2 Installation Path]\bin\win_x64\plugins\
  ```
- If the `plugins` folder doesn't exist, create it

## 🎮 Usage

### Running the Lane Assist

1. **Start Euro Truck Simulator 2**
   - Load a game or start a new delivery
   - Get on a highway or straight road for best results

2. **Run the Lane Assist Script**
   ```bash
   python play.py
   ```

3. **Controls**
   - Press **Y** to start the lane assist system
   - Press **U** to stop/pause the system
   - The system will automatically:
     - Maintain 80 km/h speed using cruise control
     - Steer the truck to stay in the center of the lane

### Collecting Training Data

If you want to train your own model with your driving style:

1. **Create Dataset Directory**
   ```bash
   mkdir dataset
   mkdir dataset/images
   ```

2. **Run the Data Collection Script**
   ```bash
   python save_dataset.py
   ```

3. **Controls During Collection**
   - Press **Q** to start recording
   - Drive normally in the game
   - Press **P** to stop recording
   - Images and telemetry data will be saved automatically

4. **Data Saved**
   - Screenshots: `dataset/images/imageXXXX.png`
   - Telemetry data: `dataset/data.txt`

### Training Your Own Model

1. **Collect sufficient data** (recommended: 10,000+ images)

2. **Run the training script**
   ```bash
   python train.py
   ```

3. **Monitor Training Progress**
   - The script uses tqdm for visual progress bars
   - Training and validation metrics are displayed after each epoch
   - Model is saved as `trained_model.keras`

4. **Update play.py** to use your trained model (if named differently)

## 📁 Project Structure

```
ETS2-Lane-Assist/
│
├── Control.py              # Low-level input control (mouse, keyboard)
├── Game.py                 # ETS2 telemetry data interface
├── play.py                 # Main script to run lane assist
├── save_dataset.py         # Collect training data while playing
├── train.py                # Train the neural network model
├── trainer_divided.py      # Alternative training script
├── dataset_merge.py        # Merge multiple datasets
│
├── ets2telemetry.py        # ETS2 telemetry data structures
├── ets2sdktelemetry.py     # ETS2 SDK telemetry interface
├── ets2sdkdata.py          # ETS2 SDK data definitions
├── sharedmemory.py         # Shared memory reader for telemetry
│
├── ets2-telemetry.dll      # ETS2 telemetry plugin (copy to game folder)
├── model.part1.rar         # Pre-trained model (part 1)
├── model.part2.rar         # Pre-trained model (part 2)
└── README.md               # This file
```

## 🧠 How It Works

### 1. Image Processing Pipeline
- Captures game screen (670x190 pixels from specific region)
- Converts to grayscale
- Applies advanced edge detection algorithm
- Resizes to 335x95 pixels for model input
- Normalizes pixel values (0-1 range)

### 2. Neural Network Architecture
The model uses a deep CNN with the following structure:
- **5 Convolutional blocks** (32, 64, 128, 256, 512 filters)
- Each block includes:
  - Conv2D layer with ReLU activation
  - Batch Normalization
  - MaxPooling (2x2)
  - Dropout for regularization
- **Dense layers**: 512 → 256 → 128 → 1 (output)
- **Output**: Single value representing steering angle (-50 to +50)

### 3. Control System
- Reads current steering position from telemetry
- Calculates required mouse movement to achieve target steering
- Uses low-level Windows API for precise input control
- Integrates with game's cruise control for speed management

### 4. Telemetry Integration
The system reads real-time data from ETS2:
- Vehicle speed (km/h, mph)
- Steering angle (user and game)
- Throttle and brake inputs
- Acceleration and rotation (X, Y, Z axes)
- Cruise control status

## 🎯 Screen Capture Configuration

The default screen capture region is defined in `Game.py`:
```python
screen_pos = (740, 440, 1410, 630)  # (left, top, right, bottom)
```

**Important**: You may need to adjust these coordinates based on:
- Your screen resolution
- Game window position
- Game resolution settings

To find the correct coordinates:
1. Run ETS2 in windowed mode
2. Use a screen coordinate tool or Python script to identify the road area
3. Update `screen_pos` in `Game.py`

## 🔍 Advanced Edge Detection

The project includes a custom edge detection algorithm:
```python
def advanced_edge_detection(image):
    # Normalize brightness
    # Adjust contrast based on image statistics
    # Apply threshold based on standard deviation
    # Returns binary edge map
```

This algorithm adapts to different lighting conditions and road types.

## 🐛 Troubleshooting

### Lane Assist Not Working
- ✅ Verify `ets2-telemetry.dll` is in the correct folder
- ✅ Check that ETS2 is running before starting `play.py`
- ✅ Ensure Python script has administrator privileges (if needed)
- ✅ Verify screen capture coordinates match your setup

### Model Performance Issues
- ✅ Collect more diverse training data (different roads, weather, time of day)
- ✅ Train for more epochs (default is 25)
- ✅ Adjust learning rate in `train.py`
- ✅ Check that input images match training data format

### DLL Not Loading
- ✅ Verify game version compatibility
- ✅ Ensure 64-bit version of the game
- ✅ Check that plugins folder exists
- ✅ Restart the game after copying DLL

### Incorrect Steering
- ✅ Calibrate the `delta` value in `play.py` (currently 0.0022968...)
- ✅ Check screen capture region is correct
- ✅ Verify steering sensitivity in game settings

## 🎓 Training Tips

1. **Data Quality**
   - Drive smoothly and consistently
   - Include various road types and conditions
   - Balance straight roads and curves
   - Avoid erratic driving during data collection

2. **Data Quantity**
   - Minimum: 5,000 images
   - Recommended: 10,000-20,000 images
   - More data = better generalization

3. **Model Tuning**
   - Monitor training/validation loss
   - Stop if validation loss increases (overfitting)
   - Adjust dropout rates if needed
   - Experiment with learning rates

## ⚠️ Disclaimer

This is an educational project demonstrating machine learning and computer vision concepts. 

- **Not for real vehicles**: This system is designed only for Euro Truck Simulator 2
- **Use at your own risk**: The AI may make mistakes
- **Game compliance**: Ensure this doesn't violate game terms of service
- **Stay engaged**: Always monitor the system while running

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share your trained models
- Improve documentation

## 📝 License

This project is open source. Please check the repository for license details.

## 🙏 Acknowledgments

- Euro Truck Simulator 2 SDK for telemetry support
- TensorFlow and Keras communities
- All contributors and testers

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Happy Trucking! 🚛💨**