import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from rembg import remove
from PIL import Image, UnidentifiedImageError

from db import Files
from app import prepare_image

# Definindo o caminho para o dataset de teste
test_image_path = 'app/image.png'  # Caminho para a imagem a ser testada

# Tamanho da imagem para entrada no modelo (deve ser o mesmo que foi usado durante o treinamento)
img_size = (150, 150)

files = Files('app/images')
files.get_only_files()

# Carregar o modelo treinado
model_path = 'model/identificador_animal.h5'  # Caminho para o arquivo do modelo treinado
model = load_model(model_path)

for path_file in files.files:
    # Preparar a imagem
    img_array = prepare_image(path_file)

    # Fazer a previsão
    predictions = model.predict(img_array)
    # Obter a classe com a maior probabilidade
    predicted_class = np.argmax(predictions, axis=1)

    # Mapear as classes para os nomes das categorias
    class_names = ['cat', 'dog', 'fish', 'horse', 'turtle']
    predicted_label = class_names[predicted_class[0]]

    print(f"{path_file} é classificada como: {predicted_label}")