import keras
import pandas as pd
from src.model import unet
from data_utils import split_data, CustomDataGenerator


def main():
    epochs = 10
    batch_size = 16
    image_size = (256, 256)

    # Load your data
    data = pd.read_csv('../data/train_ship_segmentations_v2.csv')
    # Split the data into training and testing sets
    train_data, test_data = split_data(data, empty_masks=2000, test_size=0.3)

    # Define the UNet model
    model = unet()

    # Define optimizer and loss
    optimizer = keras.optimizers.Adam(0.001)
    loss = keras.losses.BinaryCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])

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
    model.fit(train_generator, validation_data=test_generator, epochs=epochs, shuffle=True, verbose=1)

    # Save the trained model
    model.save('../models/model_v2.h5')


if __name__ == '__main__':
    main()
