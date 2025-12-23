Face Mask Detection System
Powered by Convolutional Neural Networks (CNN) & TensorFlow
A robust Deep Learning solution designed to detect the presence of face masks in images. This project leverages Computer Vision and CNN architectures to provide high-accuracy binary classification (Mask vs. No Mask).
🚀 Key Features
Dynamic Data Augmentation: Enhances model generalization by applying random rotations, zooms, and flips to the training data.
Optimized CNN Architecture: A 3-layer convolutional stack designed for efficient feature extraction from facial images.
Overfitting Protection: Implements Dropout layers and Early Stopping to ensure the model performs well on unseen data, not just the training set.
Automated Weight Recovery: Automatically restores the best performing model weights during training.
🛠️ Tech Stack
Framework: TensorFlow 2.x / Keras
Language: Python 3.x
Libraries: * NumPy: For numerical computations and array handling.
Matplotlib: For visualizing training progress.
OpenCV (Optional): For real-time image processing.
📁 Project Structure
Plaintext
├── Face_Mask_Dataset/        # Main dataset directory
│   ├── With_Mask/            # Images of people wearing masks
│   └── Without_Mask/         # Images of people without masks
├── train_model.py            # Script to build and train the CNN model
├── check_images.py           # Inference script for testing single/multiple images
├── cnn_mask_detector_v2.h5   # The final trained model file
└── README.md                 # Project documentation
⚙️ Setup & Installation
1. Requirements
Install the necessary dependencies using pip:
pip install tensorflow numpy pillow
2. Training the Model
To start the training process and generate the .h5 model file:

python train_model.py
3. Running Predictions
To test the model on new images, update the image path in check_images.py and run:
python check_images.py
🧠 Model Architecture
The network consists of:

Conv2D Layers: Three layers with 32, 64, and 128 filters respectively to capture spatial features.

MaxPooling: Downsamples the feature maps to reduce computational load.

Flatten & Dense: Converts 2D features into a 1D vector followed by a 128-unit ReLU layer.

Dropout (0.5): Prevents the model from relying on specific pixels (reducing overfitting).

Sigmoid Output: Provides a probability score between 0 and 1 for binary classification.