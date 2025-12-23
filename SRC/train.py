import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

# --- 1. Path Configuration ---
# Setting relative paths based on project structure (Data and Models are outside SRC)
dataset_path = os.path.join("..", "Data")
model_save_dir = os.path.join("..", "Models")
model_save_path = os.path.join(model_save_dir, "cnn_mask_detector_v2.h5")

# Ensure the Models directory exists
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# --- 2. Data Preparation & Augmentation ---
# Augmentation helps the model generalize better to different angles and lighting
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 # 20% of data used for validation
)

print(f"🔄 Loading data from: {os.path.abspath(dataset_path)}")

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# --- 3. CNN Model Architecture ---
model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    
    # Second Convolutional Block
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Third Convolutional Block
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Classification Head
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Prevents overfitting
    Dense(1, activation='sigmoid') # Binary output: Mask (0) or No Mask (1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 4. Training Process ---
# Stop training if validation loss doesn't improve for 3 consecutive epochs
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("🚀 Starting training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[early_stop]
)

# --- 5. Save the Model ---
model.save(model_save_path)
print(f"\n✅ Training Complete! Model saved to: {os.path.abspath(model_save_path)}")
