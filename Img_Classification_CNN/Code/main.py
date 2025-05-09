import numpy as np
import math
from data_prep import prep_and_load_data
from model import get_model
import constants as CONST
import pickle
from matplotlib import pyplot as plt
import copy
import cv2
import os
from tensorflow.keras.callbacks import TensorBoard
import time

def plotter(history_file):
    with open(history_file, 'rb') as file:
        history = pickle.load(file)

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('18_000_15epoch_accuracy.png')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('18_000_15epoch_loss.png')
    plt.show()


def video_write(model):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("./prediction.mp4", fourcc, 1.0, (400, 400))
    val_map = {1: 'Dog', 0: 'Cat'}

    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (20, 20)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

    DIR = CONST.TEST_DIR
    image_paths = os.listdir(DIR)[:200]
    count = 0
    for img_path in image_paths:
        image, image_std = process_image(DIR, img_path)
        image_std = image_std.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)

        pred = model.predict(image_std)
        arg_max = np.argmax(pred, axis=1)
        max_val = np.max(pred, axis=1)
        s = val_map[arg_max[0]] + ' - ' + str(round(max_val[0]*100, 2)) + '%'

        cv2.putText(image, s, location, font, fontScale, fontColor, lineType)
        frame = cv2.resize(image, (400, 400))
        out.write(frame)

        count += 1
        print(count)

    out.release()


def process_image(directory, img_path):
    path = os.path.join(directory, img_path)
    image = cv2.imread(path)
    image_copy = copy.deepcopy(image)

    image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
    image_std = image.astype('float32') / 255.0
    return image_copy, image_std


if __name__ == "__main__":
    print("Loading data...")
    raw_data = prep_and_load_data()

    images = [item[0] for item in raw_data]
    labels = [item[1] for item in raw_data]

    train_size = int(len(raw_data) * CONST.SPLIT_RATIO)
    train_images = np.array(images[:train_size]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
    train_labels = np.array(labels[:train_size])

    test_images = np.array(images[train_size:]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
    test_labels = np.array(labels[train_size:])

    print("Data loaded and split.")

    tensorboard = TensorBoard(log_dir=f"logs/{time.time()}")

    model = get_model()

    print('Training started...')
    history = model.fit(
        train_images, train_labels,
        batch_size=50,
        epochs=15,
        verbose=1,
        validation_data=(test_images, test_labels),
        callbacks=[tensorboard]
    )
    print('Training done.')

    model.save('18_000.h5')

    history_file = '18_000_history.pickle'
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    plotter(history_file)

    video_write(model)
