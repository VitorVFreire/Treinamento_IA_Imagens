from rembg import remove
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image

img_size = (150, 150)

def rm_background(path_file):
    try:
        with Image.open(path_file) as img:
            img_without_back = remove(img)

                # Converte para RGB se a imagem estiver em RGBA
            if img_without_back.mode == "RGBA":
                img_without_back = img_without_back.convert("RGB")

            img_without_back.save(path_file, "JPEG")
    except Exception as e:
        print(f"Erro ao remover fundo da imagem {path_file}: {e}")

# Função para carregar e preparar a imagem
def prepare_image(img_path):
    rm_background(img_path)
    # Carregar a imagem
    img = image.load_img(img_path, target_size=img_size)
    # Converter para um array numpy
    img_array = image.img_to_array(img)
    # Normalizar a imagem
    img_array = img_array / 255.0
    # Adicionar uma dimensão extra para representar o batch (modelo espera (batch_size, height, width, channels))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array