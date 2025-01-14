import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def count_class(base_path) -> int:
    caminhos = [os.path.join(base_path, nome) for nome in os.listdir(base_path)]
    return len(caminhos)-1

# Definindo o caminho para o dataset
dataset_path = "db/images"

# Parâmetros
img_size = (150, 150)  # Tamanho das imagens
batch_size = 32  # Tamanho do lote
epochs = 10  # Número de épocas para o treinamento

# Criação de geradores de imagens com aumento de dados
datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalização das imagens
    validation_split=0.2  # Usar 20% para validação
)

# Gerador para o conjunto de treinamento
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"  # Usar apenas a parte do treinamento
)

# Gerador para o conjunto de validação
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"  # Usar a parte de validação
)

# Construção do modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size[0], img_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(count_class(dataset_path), activation="softmax") 
])

# Compilando o modelo
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Early stopping para evitar overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Salvando o modelo treinado
model.save('model/identificador_animal.h5')

# Resumo do treinamento
print(f"Treinamento finalizado com {history.history['accuracy'][-1]*100:.2f}% de precisão.")
