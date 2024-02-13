import os
import cv2
import numpy as np
import pandas as pd
from keras.utils import Sequence
from sklearn.model_selection import train_test_split


train_seg = pd.read_csv('../data/train_ship_segmentations_v2.csv')
sample_submission = pd.read_csv('../data/sample_submission_v2.csv')


def rle_encode(img: np.ndarray) -> str:
    """
    Encode a binary mask represented as a 2D numpy array using Run-Length Encoding (RLE).

    Parameters:
    - img (np.ndarray): A 2D binary array representing the mask.

    Returns:
    - str: The RLE-encoded string representing the binary mask.
    """

    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle: str, shape: tuple = (768, 768)) -> np.ndarray:
    """
    Decode a Run-Length Encoded (RLE) binary mask into a 2D numpy array.

    Parameters:
    - mask_rle (str): The RLE-encoded string representing the binary mask.
    - shape (tuple, optional): The shape of the target 2D array. Default is (768, 768).

    Returns:
    - np.ndarray: A 2D binary array representing the decoded mask.
    """

    if type(mask_rle) != str:
        return np.zeros(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T  # Needed to align to RLE direction


def split_data(data: pd.DataFrame, empty_masks: int = 2000, test_size: float = 0.3, random_state: int = 42) -> tuple:
    """
    Parameters:
    - data (DataFrame): The input DataFrame containing the dataset.
    - empty_masks (int, optional): The number of images with empty masks. Defaults to 2000.
    - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.3.
    - random_state (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 42.

    Returns:
    - tuple: The training and testing sets.
    """

    masks_df = data.copy()

    # Create binary labels for the presence of ships in each image. Count the number of ships in each image.
    masks_df['ship'] = masks_df['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    masks_df['n_ships'] = masks_df.groupby('ImageId')['ship'].transform('sum')
    masks_df.drop_duplicates(subset='ImageId', keep='first', inplace=True)

    # Keep only n empty masks
    empty_masks_df = masks_df[masks_df.ship == 0]
    masks_df = masks_df[masks_df.ship == 1]
    masks_df = pd.concat([masks_df, empty_masks_df.sample(n=empty_masks, random_state=random_state)], axis=0)

    # Stratified split based on the number of ships in each image
    train_ids, test_ids = train_test_split(masks_df, test_size=test_size, stratify=masks_df['n_ships'].values,
                                           random_state=random_state)

    train_data = data[data['ImageId'].isin(train_ids.ImageId)]
    test_data = data[data['ImageId'].isin(test_ids.ImageId)]

    return train_data, test_data


class CustomDataGenerator(Sequence):
    def __init__(self, image_folder: str, csv_file: pd.DataFrame, batch_size: int = 32, image_size: tuple = (768, 768)):
        """
        Custom data generator for loading images and masks from a CSV file.

        Parameters:
        - image_folder (str): Path to the folder containing images.
        - csv_file (DataFrame): DataFrame containing image file names and corresponding encoded masks.
        - batch_size (int): Batch size for training.
        - image_size (tuple): Size of the images to be loaded.
        """
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.data = csv_file

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.

        Returns:
        - int: Number of batches.
        """
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, index: int) -> tuple:
        """
        Generates one batch of data.

        Parameters:
        - index (int): Index of the batch.

        Returns:
        - tuple: X (images) and y (masks) arrays.
        """
        batch_data = self.data[index * self.batch_size:(index + 1) * self.batch_size]

        X = []
        y = []

        for _, row in batch_data.iterrows():
            image_path = os.path.join(self.image_folder, row['ImageId'])

            # Load image
            img = cv2.imread(image_path)
            img = cv2.resize(img, self.image_size)
            img = img / 255.0  # Normalize

            # Decode RLE to mask
            all_masks = self.data[self.data['ImageId'] == row['ImageId']].EncodedPixels
            mask = np.zeros(self.image_size)
            for m in all_masks:
                decoded_mask = rle_decode(m)
                mask += cv2.resize(decoded_mask, self.image_size)

            mask = mask.astype(float)

            X.append(img)
            y.append(mask)

        return np.array(X), np.array(y)
