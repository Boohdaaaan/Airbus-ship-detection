import keras
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.model import unet
from keras import backend as K
from data_utils import split_data, CustomDataGenerator
import matplotlib.pyplot as plt

# Callbacks
reduceLROnPlate = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=1, verbose=1,
                                    mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-8)
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=5)

callbacks_list = [reduceLROnPlate, early_stopping]


# Dice coefficient
def dice_coefficient(y_true, y_pred, smooth: float = 1e-5):
    """
    Compute the Dice coefficient between two binary arrays.

    Parameters:
    - y_true (tensor): Ground truth binary tensor.
    - y_pred (tensor): Predicted binary tensor.
    - smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    - float: Dice coefficient.
    """

    # Compute the intersection of y_true and y_pred by element-wise multiplication
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    # Compute the Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

    return dice_coeff


def train(epochs: int, batch_size: int, data_path: str, image_size: tuple = (256, 256)) -> keras.callbacks.History:
    """
    Train a UNet model using provided parameters and data.

    Parameters:
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - data_path (str): Path to the CSV file containing data information.
    - image_size (tuple): Size of input images for the model. Default is (256, 256).

    Returns:
    - keras.callbacks.History: A History object containing training and validation metrics.
    """
    # Load your data
    data = pd.read_csv(data_path)

    # Split the data into training and testing sets
    train_data, test_data = split_data(data, empty_masks=2000, test_size=0.3)

    # Define the UNet model
    model = unet()

    # Define optimizer and loss
    optimizer = keras.optimizers.Adam(0.001)
    loss = keras.losses.BinaryCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=[dice_coefficient])

    # Create instances of the data generators
    train_generator = CustomDataGenerator(image_folder='../data/train_v2',
                                          csv_file=train_data,
                                          batch_size=batch_size,
                                          image_size=image_size)

    test_generator = CustomDataGenerator(image_folder='../data/train_v2',
                                         csv_file=test_data,
                                         batch_size=batch_size,
                                         image_size=image_size)

    # Train the model
    history = model.fit(train_generator, validation_data=test_generator, epochs=epochs, shuffle=True, verbose=1)

    # Save the trained model
    model.save('../models/model_v1.h5')

    return history


def plot_loss_dice(history: keras.callbacks.History, epochs: int):
    """
    Plot training and validation loss, and Dice coefficient over epochs.

    Parameters:
    - history (keras.callbacks.History): A History object containing training and validation metrics.
    - epochs (int): Number of training epochs.
    """
    plt.figure(figsize=(16, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history.history['loss'], 'bo-', label='Training loss')
    plt.plot(range(epochs), history.history['val_loss'], 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation dice coefficient
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history.history['dice_coefficient'], 'bo-', label='Training Dice Coefficient')
    plt.plot(range(epochs), history.history['val_dice_coefficient'], 'ro-', label='Validation Dice Coefficient')
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()


    # Show and save the plot
    plt.savefig('../reports/visualizations/dice_score_and_loss.png')
    plt.show()


if __name__ == '__main__':
    epochs = 20
    batch_size = 32
    data_path = '../data/train_ship_segmentations_v2.csv'

    history = train(epochs=epochs, batch_size=batch_size, data_path=data_path, image_size=(256, 256))
    plot_loss_dice(history=history, epochs=epochs)
