# Traffic Sign Recognition (GTSRB)

A Convolutional Neural Network (CNN) built using TensorFlow to classify traffic signs
from images. The model is trained on the German Traffic Sign Recognition Benchmark
(GTSRB) dataset, which contains 43 different traffic sign categories.

Images are preprocessed using OpenCV, resized to a fixed resolution, and fed into
a CNN consisting of convolutional, pooling, and fully connected layers. The network
outputs a probability distribution over all classes and predicts the most likely
traffic sign.

This project demonstrates practical computer vision techniques used in autonomous
driving systems, including image preprocessing, supervised learning, and deep learning.

## Run
pip install -r requirements.txt
python traffic.py gtsrb
