import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# -------- CONFIG --------
BASE_DIR = "../data/dataset/train"
IMG_SIZE = 64
MAX_IMAGES_PER_CLASS = 200
# -------- IMAGE LOADER --------
def load_images(folder_path, label):
    data = []
    count = 0

    for img_name in os.listdir(folder_path):
        if count >= MAX_IMAGES_PER_CLASS:
            break

        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten()
            data.append([img, label])
            count += 1
        except:
            pass

    return data
# -------- LOAD DATA --------
data = []
data += load_images(os.path.join(BASE_DIR, "cats"), 0)
data += load_images(os.path.join(BASE_DIR, "dogs"), 1)

X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

# -------- SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- TRAIN MODEL --------
model = SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)

# -------- EVALUATE --------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Task-3 Test Accuracy:", accuracy)

# -------- SAVE OUTPUT --------
results = pd.DataFrame({
    "Actual_Label": y_test,
    "Predicted_Label": y_pred
})

results.to_csv("../outputs/task3_svm_results.csv", index=False)

print("Task-3 completed. Results saved.")
