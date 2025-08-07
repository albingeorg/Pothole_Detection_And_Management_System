import numpy as np
import cv2
import glob
from keras.models import Sequential
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Global image size
size = 100

# ------------------- Load Images -------------------
def load_images_from_folder(folder_path):
    image_paths = glob.glob(folder_path + "/*.jpg") + \
                  glob.glob(folder_path + "/*.jpeg") + \
                  glob.glob(folder_path + "/*.png")
    images = []
    for path in image_paths:
        img = cv2.imread(path, 0)
        if img is not None:
            img = cv2.resize(img, (size, size))
            images.append(img)
    return np.asarray(images)

# Load training images
train_pothole = load_images_from_folder(r"C:\Users\Albin\Desktop\pt1\pothole-detection-system-using-convolution-neural-networks-master\My Dataset\train\Pothole")
train_plain = load_images_from_folder(r"C:\Users\Albin\Desktop\pt1\pothole-detection-system-using-convolution-neural-networks-master\My Dataset\train\Plain")
train_invalid = load_images_from_folder(r"C:\Users\Albin\Desktop\pt1\pothole-detection-system-using-convolution-neural-networks-master\My Dataset\train\Invalid")

# Load test images
test_pothole = load_images_from_folder(r"C:\Users\Albin\Desktop\pt1\pothole-detection-system-using-convolution-neural-networks-master\My Dataset\test\Pothole")
test_plain = load_images_from_folder(r"C:\Users\Albin\Desktop\pt1\pothole-detection-system-using-convolution-neural-networks-master\My Dataset\test\Plain")
test_invalid = load_images_from_folder(r"C:\Users\Albin\Desktop\pt1\pothole-detection-system-using-convolution-neural-networks-master\My Dataset\test\Invalid")

# ------------------- Create Datasets -------------------
# Stack train data
X_train = np.concatenate([train_pothole, train_plain, train_invalid], axis=0)
# Stack test data
X_test = np.concatenate([test_pothole, test_plain, test_invalid], axis=0)

# Create labels: 0 = Pothole, 1 = Plain, 2 = Invalid
y_train = np.concatenate([
    np.zeros(train_pothole.shape[0]),
    np.ones(train_plain.shape[0]),
    np.full(train_invalid.shape[0], 2)
])

y_test = np.concatenate([
    np.zeros(test_pothole.shape[0]),
    np.ones(test_plain.shape[0]),
    np.full(test_invalid.shape[0], 2)
])

# Shuffle and preprocess
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

X_train = X_train.reshape(X_train.shape[0], size, size, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], size, size, 1).astype('float32') / 255.0

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

# ------------------- Data Augmentation -------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.1
)

# ------------------- Define Model -------------------
def kerasModel():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(size, size, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    return model

# ------------------- Train and Evaluate -------------------
model = kerasModel()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for better convergence
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

train_gen = datagen.flow(X_train, y_train, batch_size=32, subset='training')
val_gen = datagen.flow(X_train, y_train, batch_size=32, subset='validation')

model.fit(train_gen,
          epochs=100,
          validation_data=val_gen,
          callbacks=[early_stop, lr_reduce])

# Evaluation
metrics = model.evaluate(X_test, y_test)
for name, value in zip(model.metrics_names, metrics):
    print(f"{name}: {value:.4f}")

# ------------------- Save Model -------------------
model.save("sample.h5")
