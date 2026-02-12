import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# -------- CONFIG --------
BASE_DIR = "../data/leapGestRecog/00"
IMG_SIZE = 64
MAX_IMAGES_PER_CLASS = 150


# -------- LOAD DATA --------
data = []
labels = []

gesture_folders = sorted(os.listdir(BASE_DIR))
label_map = {gesture: idx for idx, gesture in enumerate(gesture_folders)}

for gesture, label in label_map.items():
    gesture_path = os.path.join(BASE_DIR, gesture)
    count = 0

    for img_name in os.listdir(gesture_path):
        if count >= MAX_IMAGES_PER_CLASS:
            break

        img_path = os.path.join(gesture_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten()

            data.append(img)
            labels.append(label)
            count += 1
        except:
            pass


X = np.array(data)
y = np.array(labels)


# -------- TRAIN-TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------- TRAIN MODEL --------
model = SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)


# -------- EVALUATE --------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Task-4 Test Accuracy:", accuracy)


# -------- SAVE OUTPUT --------
results = pd.DataFrame({
    "Actual_Label": y_test,
    "Predicted_Label": y_pred
})

results.to_csv("../outputs/task4_gesture_results.csv", index=False)

print("Task-4 completed. Results saved.")
