# PD-Selector
An Exploration: A Clustering Algorithm Capable of Participating in Backpropagation.

*This is a clustering algorithm capable of participating in backpropagation. In application, it can replace one-hot encoding and only requires knowledge of the similarity relationship between samples to train the model. There is also room for improvement in this structure.*

## How to use

After setting up the file paths, you can directly use the GPU or CPU to run the files named ~main.py in the gaussian_sample_clustering, image_recognition, and speaker_recognition folders.

"gaussian_sample_clustering" is used for basic validation of the clustering ability of PD-Selector. "image_recognition" involves loading PD-Selector onto a simple model to verify its participation in the model's training process. "speaker_recognition" is used to validate the application of PD-Selector in more complex scenarios.

## Running image_recognition and speaker_recognition can utilize the test dataset
"FashionMNIST", "ST-AEDS-20180100_1-OS"


## Required libraries for installation
numpy,
torch,
matplotlib,
torchvision,
sys,
os
