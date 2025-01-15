import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from db import Files
from app import prepare_image

# Definindo o caminho para o modelo treinado
model_path = 'model/identificador_animal.h5'

# Carregar o modelo treinado
model = load_model(model_path)

# Carregar as imagens da pasta
files = Files('app/images')
files.get_only_files()

# Classes do modelo
names = Files('db/images')
names.load_names()
names.names.sort()
class_names = names.names

# Listas para armazenar imagens e classificações
images = []
labels = []

# Processar cada imagem
for path_file in files.files:
    try:
        # Carregar a imagem
        im = Image.open(path_file)

        # Preparar a imagem para o modelo
        img_array = prepare_image(path_file)

        # Fazer a previsão
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Obter o rótulo previsto
        predicted_label = class_names[predicted_class[0]]

        # Armazenar a imagem e o rótulo
        images.append(im)
        labels.append(predicted_label)
    except UnidentifiedImageError:
        print(f"Erro: {path_file} não é uma imagem válida.")
    except Exception as e:
        print(f"Ocorreu um erro ao processar {path_file}: {str(e)}")

# Exibir todas as imagens com suas classificações
if images:
    num_images = len(images)
    cols = 4
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(15, rows * 5))
    for i, (im, label) in enumerate(zip(images, labels)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im)
        plt.title(f"Classificação: {label}", fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Nenhuma imagem válida foi processada.")