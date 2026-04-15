import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image

# Path to dataset
data_dir = "Task3/leapGestRecog"

X, y = [], []
classes = []  # collect gesture class names

# The dataset has structure: Task3/leapGestRecog/subjectX/gestureY/*.png
for subject in os.listdir(data_dir):
    subject_folder = os.path.join(data_dir, subject)
    if not os.path.isdir(subject_folder):
        continue
    for gesture in os.listdir(subject_folder):
        gesture_folder = os.path.join(subject_folder, gesture)
        if not os.path.isdir(gesture_folder):
            continue
        if gesture not in classes:
            classes.append(gesture)
        for file in os.listdir(gesture_folder):
            img_path = os.path.join(gesture_folder, file)
            try:
                img = Image.open(img_path).convert("L")  # grayscale
                img = img.resize((64,64))
                arr = np.array(img).flatten() / 255.0
                X.append(arr)
                y.append(classes.index(gesture))
            except:
                continue

X = np.array(X)
y = np.array(y)

print("Dataset size:", X.shape, "Labels:", len(y))

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)

# Evaluation
print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=classes))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)
plt.title("Confusion Matrix Heatmap - Hand Gesture Recognition")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
