# 😷 Face Mask Detection System
> **Powered by Convolutional Neural Networks (CNN) & TensorFlow**

A robust Deep Learning solution designed to detect the presence of face masks in images. This project leverages Computer Vision and CNN architectures to provide high-accuracy binary classification (Mask vs. No Mask).

---

## 🚀 Key Features
* **Dynamic Data Augmentation:** Enhances model generalization by applying random rotations, zooms, and flips to the training data.
* **Optimized CNN Architecture:** A 3-layer convolutional stack designed for efficient feature extraction from facial images.
* **Overfitting Protection:** Implements Dropout layers and Early Stopping to ensure the model performs well on unseen data.
* **Automated Weight Recovery:** Automatically restores the best-performing model weights during training using Callbacks.

---

## 🛠️ Tech Stack
| Category | Technology |
| :--- | :--- |
| **Framework** | TensorFlow 2.x / Keras |
| **Language** | Python 3.x |
| **Libraries** | NumPy, Matplotlib, OpenCV, Pillow |

---

## 📁 Project Structure
```plaintext
├── src/
│   ├── train_model.py            # Script to build and train the CNN model
│   └── check_images.py          # Inference script for testing images
├── Face_Mask_Dataset/           # Dataset directory (With_Mask / Without_Mask)
├── cnn_mask_detector_v2.h5      # Trained model weights
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
