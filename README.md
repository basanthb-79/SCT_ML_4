# Skill Craft Internship - Task 4 

## 📌 Project Overview
This project implements a **hand gesture recognition model** using the LeapGestRecog dataset.  
The dataset contains multiple subjects performing different gestures.  
The goal is to classify gestures accurately using classical machine learning techniques without relying on TensorFlow, PyTorch, or OpenCV.

---

## 📂 Dataset Structure
The dataset is organized as follows:

```
Task3/leapGestRecog/
    subject01/
        gestureA/
            img1.png
            img2.png
            ...
        gestureB/
            ...
    subject02/
        gestureA/
        gestureB/
        ...
```

- Each **subject** folder contains multiple gesture subfolders.  
- Each **gesture** subfolder contains image files.

---

## ⚙️ Dependencies
Make sure you have the following Python libraries installed:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `Pillow`

Install them via:

```bash
pip install numpy matplotlib scikit-learn seaborn pillow
```

---

## 🧑‍💻 Code Workflow
1. **Data Loading**
   - Iterates through subjects → gestures → images.
   - Converts images to grayscale, resizes to 64×64, flattens into vectors, and normalizes.

2. **Feature Extraction**
   - Raw pixel values are used as features.

3. **Model Training**
   - Splits dataset into training (80%) and validation (20%).
   - Trains a Support Vector Machine (SVM) classifier with a linear kernel.

4. **Evaluation**
   - Generates predictions on the validation set.
   - Prints a classification report.
   - Displays a confusion matrix heatmap.

---

## ▶️ Running the Code
Run the Python script:

```bash
python gesture_recognition.py
```

Ensure the dataset path in the script is set correctly:

```python
data_dir = "Task3/leapGestRecog"
```

---

## 📊 Output
- **Classification Report**: Precision, recall, F1-score for each gesture.  
- **Confusion Matrix Heatmap**: Visual representation of prediction accuracy.  

---

## 💡 Notes
- This implementation uses classical ML (SVM) instead of deep learning.  
- You can experiment with different classifiers (e.g., RandomForest, KNN).  
- For real-time recognition, you would need to integrate webcam input and preprocess frames similarly.  

---
