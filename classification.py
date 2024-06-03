import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Carregar o modelo salvo
modelo = load_model('modelo_rede_neural.h5')


# Função para fazer previsão em uma imagem fornecida pelo usuário
def predict_user_image(image):
    # Preprocessar a imagem
    img = image.resize((64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Fazer a predição
    prediction = modelo.predict(img_array)

    # Mapear a classe prevista para o nome da classe
    class_indices = {0: 'Gato Laranja', 1: 'Gato Siamês', 2: 'Gato Sphynx'}
    predicted_class = class_indices[np.argmax(prediction)]

    # Mostrar a imagem e a predição
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()


# Função para carregar uma imagem usando um diálogo de arquivo
def load_image():
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal do Tkinter
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        predict_user_image(image)


# Executa a função de carregar a imagem
load_image()
