import tensorflow
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

binary_model = load_model('models/binary/binary_classification.h5', compile=False)
multiclass_model = load_model('models/multi/keras_model.h5', compile=False)

binaryclass_names = open("models/binary/binary_labels.txt", "r").readlines()
multiclass_names = open("models/multi/multiclass_test_labels.txt", "r").readlines()


def classification(path, model, labels):
    # Загружаем изображение
    img = image.load_img(path, target_size=(224, 224))

    # Преобразовываем изображение в массив
    img_array = image.img_to_array(img)

    # Добавляем новую "ось"
    expanded_img_array = np.expand_dims(img_array, axis=0)

    # Обработка массива numpy
    preprocessed_img = preprocess_input(expanded_img_array)

    # Запускаем предсказание модели и получаем индекс лейбла
    prediction = model.predict(preprocessed_img)
    index = prediction.argmax(axis=-1)[0]

    class_name = labels[index]

    print(f'Name: {class_name}', end='')
    print('Prediction of this picture: ' + '%.8f' % prediction[0][index] + f' Class index: {index}')
    for i in prediction[0]:
        print('%.8f' % i)

    plt.imshow(img)
    plt.xlabel(class_name + ' %.8f' % prediction[0][index])
    plt.show()


classification('data/alena(67).jpg', model=multiclass_model, labels=multiclass_names)
classification('data/arthur(12).jpg', model=multiclass_model, labels=multiclass_names)
classification('data/max(167).jpg', model=multiclass_model, labels=multiclass_names)
classification('data/kirill(17).jpg', model=multiclass_model, labels=multiclass_names)
classification('data/girl.jpg', model=multiclass_model, labels=multiclass_names)