import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Sample data from the FER-2013 dataset (grayscale 48x48 images and 7 emotion labels)
# Images are represented as flat arrays (48*48 = 2304 pixels), and labels are integer-encoded emotions.
# Normally, you would load this from a dataset, but for now, we'll use a small subset to demonstrate functionality.
sample_train_images = np.array([
    np.random.randint(0, 255, (48, 48), dtype=np.uint8),  # Random image data
    np.random.randint(0, 255, (48, 48), dtype=np.uint8),  # Random image data
    np.random.randint(0, 255, (48, 48), dtype=np.uint8)   # Random image data
])

sample_train_labels = np.array([0, 1, 2])  # Example labels: 0, 1, and 2 representing emotions

sample_test_images = np.array([
    np.random.randint(0, 255, (48, 48), dtype=np.uint8),  # Random image data
    np.random.randint(0, 255, (48, 48), dtype=np.uint8)   # Random image data
])

sample_test_labels = np.array([1, 2])  # Example labels: 1 and 2 representing emotions

# Reshape the data to add the channel dimension (grayscale images)
sample_train_images = np.expand_dims(sample_train_images, axis=-1)
sample_test_images = np.expand_dims(sample_test_images, axis=-1)

# Normalize the images
sample_train_images = sample_train_images / 255.0
sample_test_images = sample_test_images / 255.0

# Step 3: Define the CNN model architecture
def create_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(7, activation='softmax')  # 7 emotions in FER-2013 dataset
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 4: Create the model
model = create_model()

# Step 5: Train the model on the sample data
model.fit(sample_train_images, sample_train_labels, epochs=5, validation_data=(sample_test_images, sample_test_labels))

# Step 6: Save the model as an .h5 file
model.save('emotion_model_sample.h5')

print("Model training complete and saved as 'emotion_model_sample.h5'.")
