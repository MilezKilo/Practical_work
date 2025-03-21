# ----------------------------------------
#           LIBRARIES IMPORT
# ----------------------------------------

from keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import keras

# Загружаем модель
multi_model = load_model("models/multi/keras_model.h5", compile=False)
# binary_model = load_model("models/binary/binary_classification.h5", compile=False)

face_detect = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_COMPLEX

# Загружаем метки
# binaryclass_names = open("models/binary/binary_labels.txt", "r").readlines()
multiclass_names = open("models/multi/multiclass_test_labels.txt", "r").readlines()


# Захватываем видео с камеры
camera = cv2.VideoCapture(0)


# Метод бинарной классификации (всего 2 класса)
def binary_model_detector():
    while True:
        # Захватываем изображение с видео
        ret, original_image = camera.read()
        faces = face_detect.detectMultiScale(original_image, 1.3, 5)

        for x, y, w, h in faces:
            # Ресайз изображения для предугадывания моделью
            img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_AREA)
            img = (np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3) / 127.5) - 1

            # Предугадывание натренированной модели
            prediction = binary_model.predict(img)

            # Индекс лейблов
            index = prediction.argmax(axis=-1)[0]

            print(index)
            for i in prediction[0]:
                print('%.10f' % i, end=' ')

            # Данные для вывода на экран
            class_name = binaryclass_names[index]
            confidence_score = prediction[0][index]

            # Добавляем рамку определения лица
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Добавляем на экран вероятность предсказания модели
            cv2.putText(original_image,
                        str(np.round(confidence_score * 100))[:-2] + '%' + class_name,
                        (180, 75),
                        font,
                        0.75,
                        (255, 0, 0),
                        2, cv2.LINE_AA)
        # Включаем камеру
        cv2.imshow('camera', original_image)

        keyboard_input = cv2.waitKey(1)

        # Нажмите ESC для выхода из цикла
        if keyboard_input == 27:
            break

    # "Освобождаем камеру
    camera.release()
    cv2.destroyAllWindows()


# Метод мультиклассовой классификации (5 классов)
def multiclass_model_detector():
    while True:
        # Захватываем изображение с видео и определяем лицо
        ret, original_image = camera.read()
        faces = face_detect.detectMultiScale(original_image, 1.3, 5)

        for x, y, w, h in faces:
            # Ресайз и сохранение изображения
            try:
                crop_img = original_image[y - 65:y + h + 65, x - 65:x + w + 65]
                # crop_img = original_image[y:y + h, x:x + w]
                img = cv2.resize(crop_img, (224, 224))
                cv2.imwrite('data/face_detected.jpg', img)

                # Загрузка и преобразовывание изображение в удобный для модели формат
                loaded_image = image.load_img('data/face_detected.jpg', target_size=(224, 224))
                img_array = image.img_to_array(loaded_image)
                expanded_img_array = np.expand_dims(img_array, axis=0)

                # Обработка массива numpy
                preprocessed_img = preprocess_input(expanded_img_array)

                # Предугадывание натренированной модели
                prediction = multi_model.predict(preprocessed_img)

                # Индекс лейблов
                index = prediction.argmax(axis=-1)[0]

                # Данные для вывода на экран
                class_name = multiclass_names[index]
                confidence_score = prediction[0][index]

                print(f'Name: {class_name}')
                print('Prediction of this picture: ' + '%.8f' % prediction[0][index] + f' Class index: {index}')
                for i in prediction[0]:
                    print('%.8f' % i, end=' ')

                # Добавляем рамку определения лица
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

                # Добавляем на экран вероятность предсказания модели
                cv2.putText(original_image,
                            class_name + ' ' + str(np.round(confidence_score * 100))[:-2] + '%',
                            (180, 75),
                            font,
                            0.75,
                            (255, 0, 0),
                            2, cv2.LINE_AA)
            except cv2.error as e:
                pass

        # Включаем камеру
        cv2.imshow('camera', original_image)

        keyboard_input = cv2.waitKey(1)

        # Нажмите ESC для выхода из цикла
        if keyboard_input == 27:
            break

    # "Освобождаем камеру
    camera.release()
    cv2.destroyAllWindows()


multiclass_model_detector()
