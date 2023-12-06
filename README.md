# Airbus Ship Detection Challenge

Welcome to my repository for the Airbus Ship Detection Kaggle competition! In this competition, the goal is to build a model that efficiently detects ships in satellite images, even in challenging conditions such as cloud cover or haze. The competition is sponsored by Airbus, aiming to address the growing concerns related to maritime activities and enhance monitoring capabilities for various organizations.

## Repository Structure

This repository contains the following folders:

1. **Data**: This folder includes examples (10 images from the dataset), `train_ship_segmentations_v2.csv`, and `sample_submission_v2.csv`.

2. **Models**: The `Models` folder contains the trained UNet model (`seg_model.h5`), which is the core component of the ship detection solution. *Important:* I was unable to train my own model due to a lack of time, so I decided to use a pre-trained model for now. Later, I plan to train my own model and replace the existing one.

4. **Notebooks**: Explore the provided Jupyter notebooks:
   - `EDA.ipynb`: Exploratory Data Analysis notebook.
   - `inference.ipynb`: Inference notebook for making and visualizing predictions. 

5. **src**: This directory contains the source code for various utility functions and scripts:
   - `data_utils.py`: Functions such as `rle_encode`, `rle_decode`, `split_data`, and `CustomDataGenerator`.
   - `model.py`: The UNet architecture for ship detection.
   - `train.py`: Script for model training and saving the trained model.
   - `inference.py`: Script for loading the trained model, making predictions, and generating a submission file.
   
## Detailed Breakdown of .py Files
#### 1. Data Utilities (`data_utils.py`)

The `data_utils.py` module offers a suite of utility functions crucial for data preprocessing and manipulation. Notable functions include:

- **RLE Encoding and Decoding:** The `rle_encode` and `rle_decode` functions facilitate the conversion between binary masks and Run-Length Encoded (RLE) strings, a pivotal aspect of the competition's evaluation metric.

- **Data Splitting:** The `split_data` function intelligently stratifies the dataset, ensuring a balanced distribution of ship instances across training and testing sets.

- **Custom Data Generator:** The `CustomDataGenerator` class inherits from Keras' Sequence class, providing an efficient data generator for model training.

#### 2. UNet Model Architecture (`model.py`)

The heart of the solution lies in the definition of the UNet architecture for semantic segmentation. The `unet` function in `model.py` crafts a convolutional neural network with an encoder-decoder structure, capable of effectively capturing intricate patterns in satellite images.

The UNet model comprises encoder layers for feature extraction, a middle layer for latent representation, and decoder layers for precise segmentation. The architecture is tailored to the specific challenges posed by maritime satellite imagery, ensuring a powerful and adaptive solution.

#### 3. Model Training (`train.py`)

The training script, `train.py`, orchestrates the entire training process. Key steps include:

- **Data Loading and Splitting:** The script loads the dataset, where the `split_data` function separates images into training and testing sets, considering the presence of ships in each image.

- **Model Compilation and Training:** Using the UNet model defined in `model.py`, the script compiles the model with an Adam optimizer and binary crossentropy loss. The model is then trained using a custom data generator, promoting efficiency and scalability.

- **Save Model:** After successful training, the script saves the trained model in the `../models` directory, ensuring easy retrieval for future use or inference.

#### 4. Inference and Submission (`inference.py`)

The `inference.py` script takes the trained UNet model and applies it to new data for ship detection. The process involves:

- **Preprocessing Input Images:** Images are loaded, converted to RGB, resized, and normalized, preparing them for model inference.

- **Prediction and Morphological Operations:** The model predicts ship masks, and morphological operations are applied to distinguish individual objects in the segmentation.

- **RLE Encoding for Submission:** The resulting segmentation masks are RLE encoded, and a submission file is generated, adhering to  thecompetition's submission requirements.

## Getting Started

To get started with this repository, follow these steps:

1. Clone the repository to your local machine.
2. Explore the provided data and notebooks.
3. Train the UNet model using the `train.py` script.
4. Use the trained model for inference and generate predictions with the `inference.py` script.

Feel free to reach out if you have any questions or suggestions. Happy coding!

**Note**: Ensure you have the required dependencies installed, and refer to the documentation within each script for more detailed instructions.

## Conclusion

This solution provides a comprehensive, end-to-end pipeline for ship detection in satellite images. From robust data preprocessing and a tailored UNet architecture to efficient model training and submission generation, every aspect is meticulously crafted to address the challenges posed by the Airbus Ship Detection Kaggle competition.
