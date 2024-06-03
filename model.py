import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Diretórios de treinamento e teste
train_dir = './training_set'
test_dir = './test_set'

# Preprocessamento de dados
gerador_treinamento = ImageDataGenerator(rescale=1. / 255, rotation_range=7, horizontal_flip=True, zoom_range=0.2)
base_treinamento = gerador_treinamento.flow_from_directory(train_dir, target_size=(64, 64), batch_size=8,
                                                           class_mode='categorical')

gerador_teste = ImageDataGenerator(rescale=1. / 255)
base_teste = gerador_teste.flow_from_directory(test_dir, target_size=(64, 64), batch_size=8, class_mode='categorical',
                                               shuffle=False)

# Definição da rede neural
rede_neural = Sequential()
rede_neural.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2, 2)))

rede_neural.add(Conv2D(32, (3, 3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2, 2)))

rede_neural.add(Flatten())

rede_neural.add(Dense(units=128, activation='relu'))
rede_neural.add(Dense(units=64, activation='relu'))
rede_neural.add(Dense(units=3, activation='softmax'))

# Compilação e treinamento do modelo
rede_neural.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
rede_neural.fit(base_treinamento, epochs=200, validation_data=base_teste)

# Avaliação do modelo
previsoes = rede_neural.predict(base_teste)
previsoes2 = np.argmax(previsoes, axis=1)
accuracy = accuracy_score(previsoes2, base_teste.classes)
print(f'Accuracy: {accuracy}')

# Matriz de Confusão
cm = confusion_matrix(base_teste.classes, previsoes2)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Função para fazer previsão em uma imagem aleatória do conjunto de teste
def predict_random_image():
    class_indices = base_treinamento.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}

    random_class = random.choice(list(class_indices.keys()))
    random_image_path = random.choice(os.listdir(os.path.join(test_dir, random_class)))
    random_image_full_path = os.path.join(test_dir, random_class, random_image_path)

    img = load_img(random_image_full_path, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = rede_neural.predict(img_array)
    predicted_class = idx_to_class[np.argmax(prediction)]

    # Mostra a imagem escolhida
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f'Chosen Image: {random_image_path}')
    plt.axis('off')
    plt.show()

    # Mostra a predição
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()


predict_random_image()
rede_neural.save('modelo_rede_neural.h5')
