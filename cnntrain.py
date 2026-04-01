import os
import pandas as pd
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Parameters
# -----------------------------
data_dir = "train"  # Update with your data directory
input_shape = (50, 50, 3)
test_size = 0.2

# -----------------------------
# Load dataset
# -----------------------------
subfolders = os.listdir(data_dir)
data = []

for cls in subfolders:
    cls_dir = os.path.join(data_dir, cls)
    for img_name in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, img_name)
        data.append((img_path, cls))

train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
train_df = pd.DataFrame(train_data, columns=["image_path", "class"])
test_df = pd.DataFrame(test_data, columns=["image_path", "class"])

# -----------------------------
# Image preprocessing function
# -----------------------------
def preprocess_image(img_path):
    try:
        img_array = cv2.imread(img_path, 1)
        img_array = cv2.medianBlur(img_array, 1)
        new_array = cv2.resize(img_array, input_shape[:2])
        return new_array
    except Exception as e:
        return None

# -----------------------------
# Prepare training data
# -----------------------------
training_data = []

for index, row in train_df.iterrows():
    img_path = row["image_path"]
    class_num = subfolders.index(row["class"])
    img_data = preprocess_image(img_path)
    if img_data is not None:
        training_data.append([img_data, class_num])

random.shuffle(training_data)

X = np.array([features for features, label in training_data]*3)
y = np.array([label for features, label in training_data]*3)

# Normalize images
X = X / 255.0

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(len(subfolders)))  # output layer for number of classes
model.add(Activation("softmax"))

# Compile CNN
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train CNN
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_split=0.2)

# Evaluate CNN
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'CNN Training Accuracy: {train_acc*100:.2f}%')
print(f'CNN Testing Accuracy: {test_acc*100:.2f}%')

# Save CNN model
model.save('CNN.model')

# -----------------------------
# Plot CNN Training History
# -----------------------------
plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("CNN Loss")
plt.legend()
plt.show()

# -----------------------------
# CNN Confusion Matrix
# -----------------------------
y_cnn_pred_prob = model.predict(X_test)
y_cnn_pred = np.argmax(y_cnn_pred_prob, axis=1)

print("\nCNN Classification Report\n")
print(classification_report(y_test, y_cnn_pred))

cm_cnn = confusion_matrix(y_test, y_cnn_pred)
plt.figure(figsize=(8,8))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CNN Confusion Matrix")
plt.show()

