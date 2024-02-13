import os
import cv2
import numpy as np
import pandas as pd
import gdown
import keras
from skimage.measure import label, regionprops
from data_utils import rle_encode
from train import dice_coefficient


# Load and preprocess the input image
def preprocess_input(image_path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Preprocesses an image for input to a model.

    Parameters:
    - image_path (str): Path to the image file.
    - target_size (tuple): Target size for resizing the image.

    Returns:
    - np.ndarray: Preprocessed image.
    """
    # Read the image from the specified path
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image from {image_path}")

    # Check if the loaded image has valid dimensions
    if image.shape[0] == 0 or image.shape[1] == 0:
        print("Error: Loaded image has invalid dimensions.")

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image


# Load the model with custom_objects parameter
def load_model(model_url: str, model_path: str) -> keras.models.Model:
    """
    Loads a model from a given URL and saves it to a specified output path.

    Parameters:
    - url (str): The URL from which the model should be downloaded.
    - model_path (str): The path where the downloaded model should be saved.

    Returns:
    - keras.models.Model: The loaded Keras model.
    """
    # If the output path doesn't exist, download the model from the URL and save it to the output path
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=True, fuzzy=True)

    # Register the custom loss function and load model
    keras.losses.Dice = dice_coefficient
    keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient
    loaded_model = keras.models.load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})

    return loaded_model


# Make submission file
def make_submission(path_to_folder: str, model: keras.models.Model) -> pd.DataFrame:
    """
    Generate a submission DataFrame containing image IDs and corresponding encoded pixels.

    Parameters:
    - path_to_folder (str): Path to the folder containing images.
    - model (keras.models.Model): The trained Keras model used for prediction.

    Returns:
    - pd.DataFrame: Submission DataFrame with columns 'ImageId' and 'EncodedPixels'.
    """
    # Get a list of image filenames in the specified folder
    list_of_images = os.listdir(path_to_folder)

    # Initialize lists to store image IDs and corresponding encoded pixels
    image_id = []
    encoded_pixels = []

    # Iterate over each image
    for img_name in list_of_images:
        # Obtain the model prediction for the current image
        img = preprocess_input(os.path.join(path_to_folder, img_name))
        mask = model.predict(img, verbose=0)
        mask = np.squeeze(mask, axis=(0, 3))
        mask = cv2.resize(mask, (768, 768))
        mask = (mask > 0.3).astype(int)

        if np.all(mask == 0):
            # If no objects are detected in the mask, add empty values to the lists
            image_id.append(img_name)
            encoded_pixels.append('')
        else:
            # Apply morphological operation to distinguish individual objects
            labeled_mask = label(mask)
            for region in regionprops(labeled_mask):
                # Create a mask for the current object
                single_ship_mask = (labeled_mask == region.label).astype(np.uint8)

                # Obtain Run-Length Encoding (RLE) for the mask
                rle = rle_encode(single_ship_mask)

                # Add image ID and encoded pixels to the lists
                image_id.append(img_name)
                encoded_pixels.append(rle)

    # Create a DataFrame from the lists
    df = pd.DataFrame({"ImageId": image_id, "EncodedPixels": encoded_pixels})
    return df


# Save predicted masks for images in a folder
def save_predicted_masks(images_folder: str, masks_folder: str, model: keras.models.Model):
    """
    Predict masks for images in the given folder using the provided model and save the masks.

    Parameters:
    - images_folder (str): Path to the folder containing input images.
    - masks_folder (str): Path to the folder where predicted masks will be saved.
    - model (keras.models.Model): The trained Keras model used for prediction.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)

    # List all image files in the input folder
    images = os.listdir(images_folder)

    # Iterate over each image file
    for img_name in images:
        # Preprocess the input image
        img_path = os.path.join(images_folder, img_name)
        image = preprocess_input(img_path)

        # Predict the mask for the image
        output_mask = model.predict(image, verbose=0)

        # Convert the output mask to an image format supported by OpenCV
        output_mask = np.squeeze(output_mask)
        output_mask = (output_mask > 0.3).astype(int)
        output_mask = (output_mask * 255).astype(np.uint8)

        # Save the output mask
        output_mask_path = os.path.join(masks_folder, img_name)
        cv2.imwrite(output_mask_path, output_mask)


if __name__ == "__main__":
    model_url = 'https://drive.google.com/uc?export=download&id=1eyKPgbBQ0bY4d5DIX1aW_Y2auoEP1SJr'
    model_path = "../models/model_v1.h5"
    folder_path = "../data/examples"
    masks_folder = '../data/predicted_masks'

    # Load model
    model_unet = load_model(model_url, model_path)

    # Make submission file (for 11 images)
    submission = make_submission(folder_path, model_unet)
    submission.to_csv('../data/submission.csv', index=False)

    # Save predicted masks for images in a folder
    save_predicted_masks(folder_path, masks_folder, model_unet)
