import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

TRAIN_DATA_DIR = '../human_detector/data/train'
VALIDATION_DATA_DIR = '../human_detector/data/val'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DATA_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical')

model = Sequential([
    Conv2D(32, (3, 3), 1,  activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), 1, activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), 1, activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

history = model.fit(train_generator,
                    steps_per_epoch=500 // 32,
                    validation_data=validation_generator,
                    validation_steps=500 // 32,
                    epochs=11, callbacks=[early_stopping, reduce_lr])

model.save('test.h5')


def show_loses_accuracy():
    history_dict = history.history

    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]

    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]

    epochs = range(1, len(loss_values) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, "bo", label="Потери на этапе обучения")
    plt.plot(epochs, val_loss_values, "b", label="Потери на этапе проверки")
    plt.title("Потери на этапах обучения и проверки")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, "bo", label="Точность на этапе обучения")
    plt.plot(epochs, val_acc, "b", label="Точность на этапе проверки")
    plt.title("Точность на этапах обучения и проверки")
    plt.xlabel("Эпохи")
    plt.ylabel("Точность")
    plt.legend()

    plt.show()
