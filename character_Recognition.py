import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. KAGGLE CSV LOADING & PREPROCESSING
# ==========================================
train_path = "emnist-balanced-train.csv"
test_path = "emnist-balanced-test.csv"

# This block makes sure your files are in the right folder!
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError(
        f"❌ Missing CSV files in your workspace! "
        f"Please ensure '{train_path}' and '{test_path}' are placed right inside: {os.getcwd()}"
    )

print("🔄 Reading Kaggle CSV files (This might take 10-20 seconds)...")
train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)

print("🧹 Extracting features and labels...")
y_train = train_data.iloc[:, 0].values
X_train = train_data.iloc[:, 1:].values

y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values

# Reshape flat array maps back into 28x28 image frames
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

print("🔄 Adjusting image orientations...")
# Fixing the native EMNIST reflection/rotation issue
X_train = np.array([np.rot90(np.flipud(img)) for img in X_train])
X_test = np.array([np.rot90(np.flipud(img)) for img in X_test])

# Normalize pixel values [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add a single grayscale channel dimension for the CNN network format
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

num_classes = 47 
print(f"📊 Dataset successfully structured: {X_train.shape[0]} samples processed.")

# One-hot encoding mapping targets
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

class_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# ==========================================
# 2. BUILDING THE CNN ARCHITECTURE
# ==========================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

print("\n🖥️ Model Setup Complete:")
model.summary()

# ==========================================
# 3. COMPILING AND TRAINING
# ==========================================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n🚀 Commencing model training...")
history = model.fit(X_train, y_train_cat, 
                    epochs=8, 
                    batch_size=128, 
                    validation_data=(X_test, y_test_cat))

# ==========================================
# 4. EVALUATION & CRITICAL SAVING STEP
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n🎯 Training Completed. Test Accuracy: {test_acc * 100:.2f}%")

# 💾 MOVE SAVING HERE (Right after training finishes!)
print("\n💾 Writing 'character_model.h5' to workspace...")
model.save("character_model.h5")
print("✅ File successfully saved! You can now launch app.py safely.")

# ==========================================
# 5. VISUAL PLOTTING (Can be safely closed now)
# ==========================================
print("\n📊 Launching preview plot window...")
predictions = model.predict(X_test[:5])
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    pred_char = class_mapping[np.argmax(predictions[i])]
    true_char = class_mapping[y_test[i]]
    plt.title(f"Pred: {pred_char}\nTrue: {true_char}", color='green' if pred_char == true_char else 'red')
    plt.axis('off')
plt.tight_layout()
plt.show() # If you exit here now, your model is already safely saved!
# ==========================================
# SAVE THE TRAINED MODEL
# ==========================================
print("\n💾 Saving model weights to disk...")

# Option A: Standard Native Keras format (Highly recommended for newer TensorFlow)
model.save("character_model.keras") 

# Option B: Legacy HDF5 format (If your app.py specifically looks for .h5)
model.save("character_model.h5")

print("✅ Model file generated successfully in your project folder!")

