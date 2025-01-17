from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from db import Files
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

img_height, img_width = 150, 150

# Função para prever uma imagem
def predict_image(class_name, image_path, model, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    print(f'{image_path} - {class_name}: {prediction[0]}')

    if prediction[0] >= 0.5:
        return class_name, prediction[0], True
    else:
        return f"no_{class_name}", prediction[0], False

# Carregar as classes
classes_names = Files('db/images')
classes_names.load_names()

# Carregar os modelos
models = {class_name: load_model(f'model/{class_name}_binary_best.keras') for class_name in classes_names.names}

# Processar imagens
files = Files('app/images')
files.get_only_files()

images = []
labels = []

for path_file in files.files:
    try:
        # Carregar e classificar a imagem
        im = Image.open(path_file)
        final_label = 'Not Found'
        for class_name in classes_names.names:
            label, confidence, result_predict = predict_image(class_name, path_file, models[class_name], img_height, img_width)
            if result_predict:
                final_label = label
                break

        # Armazenar a imagem e seu rótulo
        images.append(im)
        labels.append(final_label)
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

