import os
import cv2
import numpy as np
import pandas as pd
import keras
import keras.backend as K
from skimage.measure import label, regionprops
from data_utils import rle_encode


# Load and preprocess the input image
def preprocess_input(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Define IoU function
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1 - y_true, 1 - y_pred)  # empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return -K.mean((intersection + eps) / (union + eps), axis=0)


# Register IoU function
keras.losses.IoU = IoU


# Load the model with custom_objects parameter
def load_model(model_path):
    loaded_model = keras.models.load_model(model_path, custom_objects={'IoU': IoU})
    return loaded_model


# Make submission file
def make_submission(folder_path):
    list_of_images = os.listdir(folder_path)
    image_id = []
    encoded_pixels = []

    for img_name in list_of_images:
        # Obtaining the model prediction.
        img = preprocess_input(os.path.join(folder_path, img_name, ))
        mask = model.predict(img, verbose=0)
        mask = np.squeeze(mask, axis=(0, 3))
        mask = cv2.resize(mask, (768, 768))

        # Applying the morphological operation to distinguish individual objects
        labeled_mask = label(mask)

        for region in regionprops(labeled_mask):
            # Creating a mask for the current object
            single_ship_mask = (labeled_mask == region.label).astype(np.uint8)

            # Obtaining RLE for the mask
            rle = rle_encode(single_ship_mask)

            # Adding values to the lists
            image_id.append(img_name)
            encoded_pixels.append(rle)

    # Creating a DataFrame.
    df = pd.DataFrame({"ImageId": image_id, "EncodedPixels": encoded_pixels})
    return df


if __name__ == "__main__":
    model_path = "../models/seg_model.h5"
    folder_path = "../data/examples"

    # Load model
    model = load_model(model_path)

    # Make submission file
    submission = make_submission(folder_path)
    # submission.to_csv('../data/submission.csv', index=False)

