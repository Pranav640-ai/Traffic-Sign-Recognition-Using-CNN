import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")
    images, labels = load_data(sys.argv[1])
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS)
    model.evaluate(x_test,  y_test, verbose=2)
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
def load_data(data_dir):
    files = Path(data_dir).iterdir()
    labels = []
    images = []

    for file in files:
        if not file.is_dir():
            continue
        for img in file.iterdir():
            if not img.is_file():
                continue
            im = cv2.imread(str(img))
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT))
            images.append(resized)
            labels.append(int(file.name))
    return images, labels


def get_model():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                40, (3, 3), activation="relu",
                input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    main()
