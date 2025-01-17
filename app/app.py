from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from db import Files
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from numpy import expand_dims, argmax

img_height, img_width = 100, 100

# Função para prever uma imagem
def predict_image(image_path, model, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)

    # Corrigido: usar img_to_array para converter a imagem corretamente
    y = model.predict(np.expand_dims(img_to_array(img), axis=0))
    return argmax(y)

# Carregar as classes
classes_names = Files('db/images')
classes_names.load_names()
classes_names.names.sort()
classes_names = classes_names.names

# Carregar os modelos
model = load_model('model/model.keras')

# Processar imagens
files = Files('app/images')
files.get_only_files()

images = []
labels = []

for path_file in files.files:
    try:
        # Carregar e classificar a imagem
        im = Image.open(path_file)
        indice = predict_image(path_file, model, img_height, img_width)

        # Armazenar a imagem e seu rótulo
        images.append(im)
        labels.append(classes_names[indice])
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
