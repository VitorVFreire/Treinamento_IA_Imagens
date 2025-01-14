import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definindo o caminho para o dataset de teste
test_image_path = 'app/image.png'  # Caminho para a imagem a ser testada

# Tamanho da imagem para entrada no modelo (deve ser o mesmo que foi usado durante o treinamento)
img_size = (150, 150)

# Carregar o modelo treinado
model_path = 'model/identificador_animal.h5'  # Caminho para o arquivo do modelo treinado
model = load_model(model_path)

# Função para carregar e preparar a imagem
def prepare_image(img_path):
    # Carregar a imagem
    img = image.load_img(img_path, target_size=img_size)
    # Converter para um array numpy
    img_array = image.img_to_array(img)
    # Normalizar a imagem
    img_array = img_array / 255.0
    # Adicionar uma dimensão extra para representar o batch (modelo espera (batch_size, height, width, channels))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Preparar a imagem
img_array = prepare_image(test_image_path)

# Fazer a previsão
predictions = model.predict(img_array)
# Obter a classe com a maior probabilidade
predicted_class = np.argmax(predictions, axis=1)

# Mapear as classes para os nomes das categorias
class_names = ['cat', 'dog', 'fish', 'horse', 'turtle']  # Atualize conforme suas classes
predicted_label = class_names[predicted_class[0]]

print(f"A imagem é classificada como: {predicted_label}")