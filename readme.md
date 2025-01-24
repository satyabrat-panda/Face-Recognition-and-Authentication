# Face Recognition System

This repository contains a Python-based face recognition system that includes functionalities for data collection, model training, and face recognition testing.

## Features
- **Data Collection**: Captures images of faces and saves them in a dataset folder for training.
- **Training**: Trains a face recognition model using the LBPH algorithm.
- **Recognition**: Recognizes faces in real-time using the trained model.

## Project Structure
```
.
├── datacollect.py    # Script for collecting face images and storing them in datasets.
├── training.py       # Script for training the face recognition model using collected data.
├── test.py           # Script for testing the recognition of faces in real-time.
├── Trainer.yml       # Trained model file (generated after running training.py).
├── names.txt         # List of names associated with the trained IDs (generated after running training.py).
└── datasets/         # Directory where face images are stored (created automatically).
```

## Prerequisites
- Python 3.x
- OpenCV
- NumPy
- PIL (Pillow)

Install the required libraries using pip:
```bash
pip install opencv-python opencv-contrib-python numpy pillow
```

## Usage

### 1. Data Collection
Run the `datacollect.py` script to capture face images. Enter the person's name when prompted. Images will be saved in the `datasets/` directory.
```bash
python datacollect.py
```

### 2. Training
Once data collection is complete, run the `training.py` script to train the face recognition model. This will generate:
- `Trainer.yml`: The trained model file.
- `names.txt`: A file mapping IDs to names.
```bash
python training.py
```

### 3. Face Recognition
Run the `test.py` script to recognize faces in real-time using your webcam.
```bash
python test.py
```

## How It Works
- **Data Collection**: Captures images of a person's face and saves them in grayscale format in a folder named after the person.
- **Training**: Uses the LBPH (Local Binary Patterns Histograms) algorithm to train a face recognition model based on the collected data.
- **Testing**: Detects faces in real-time and predicts their identity using the trained model. If a match is not found, it marks the face as "Unknown."

## Notes
- Make sure your webcam is properly connected.
- Adjust the confidence threshold in `test.py` if needed for better accuracy.
- To stop any script, press `q`.

## Author
Satyabrat Panda  
GitHub Profile : satyabrat-panda

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
