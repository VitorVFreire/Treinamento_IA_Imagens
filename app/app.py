from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from db import Files
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

img_height, img_width = 50, 50

# Função para prever uma imagem
def predict_image(image_path, model, img_height, img_width, class_names, confidence_threshold=0.5):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar a previsão
    predictions = model.predict(img_array)
    predicted_confidence = np.max(predictions)  # Confiança da classe mais provável
    predicted_class = np.argmax(predictions)  # Índice da classe mais provável
    # Verificar se a confiança atinge o limiar
    if predicted_confidence >= confidence_threshold:
        return class_names[predicted_class]
    else:
        return "não identificado"

# Carregar as classes
class_names = Files('db/images', ['train', 'test'])
class_names.load_names()
class_names.names

# Carregar os modelos
model = load_model('model/model.keras')

# Processar imagens
files = Files('app/images', ['train', 'test'])
files.get_only_files()

images = []
labels = []

for path_file in files.files:
    try:
        # Carregar e classificar a imagem
        im = Image.open(path_file)
        resultado = predict_image(path_file, model, img_height, img_width, class_names.names, confidence_threshold=0.5)

        # Armazenar a imagem e seu rótulo
        images.append(im)
        labels.append(resultado)
    except UnidentifiedImageError:
        print(f"Erro: {path_file} não é uma imagem válida.")
    except Exception as e:
        print(f"Ocorreu um erro ao processar {path_file}: {str(e)}")

# Exibir imagens com rótulos
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
