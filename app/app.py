from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_height, img_width = 150, 150
# Carregar o modelo treinado
model = load_model('model/my_model.keras')

# Função para prever uma imagem
def predict_image(image_path, model, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Dog", prediction[0]
    else:
        return "Cat", prediction[0]

# Exemplo de previsão
image_path = "app/images/gato.png"
label, confidence = predict_image(image_path, model, img_height, img_width)
print(f"Predição: {label} com confiança de {confidence}")
