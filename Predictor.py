import numpy as np
import cv2
import glob
from keras.models import load_model
from tensorflow.keras.utils import to_categorical  # Updated here

# Global size
size = 100

# Load the trained model
model = load_model(r'C:\Users\Albin\Downloads\pt1\sample.h5')

# Load a non-pothole test image
nonPotholeTestImages = glob.glob(
    r"C:\Users\Albin\Downloads\pt1\pothole-detection-system-using-convolution-neural-networks-master\My Dataset\test\Plain\3.jpg"
)
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]
test2 = [cv2.resize(img, (size, size)) for img in test2 if img is not None]
temp4 = np.asarray(test2)

# Load a pothole test image
potholeTestImages = glob.glob(
    r"C:\Users\Albin\Downloads\pt1\pothole-detection-system-using-convolution-neural-networks-master\My Dataset\test\Pothole\2.jpg"
)
test1 = [cv2.imread(img, 0) for img in potholeTestImages]
test1 = [cv2.resize(img, (size, size)) for img in test1 if img is not None]
temp3 = np.asarray(test1)

# Combine and preprocess test data
X_test = []
X_test.extend(temp3)
X_test.extend(temp4)
X_test = np.asarray(X_test).reshape(-1, size, size, 1)

# Labels: 1 = pothole, 0 = plain
y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)

y_test = np.array(list(y_test1) + list(y_test2))
y_test = to_categorical(y_test)

# Make predictions (replacing deprecated predict_classes)
predictions = np.argmax(model.predict(X_test), axis=1)

# Print results
for i in range(len(X_test)):
    label = "Pothole" if predictions[i] == 1 else "Plain"
    print(f">>> Predicted={label} ({predictions[i]})")
