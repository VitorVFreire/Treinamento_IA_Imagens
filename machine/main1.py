import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from db import Files

# Classes do modelo
files = Files('db/images')
count_files = int(files.count_files())

def count_class(base_path) -> int:
    caminhos = [os.path.join(base_path, nome) for nome in os.listdir(base_path)]
    return len(caminhos) - 1

# Caminho do dataset
dataset_path = "db/images"
print(f'Count Images: {count_files} | Count Class: {count_class(dataset_path)}')

# Parâmetros
img_size = (224, 224)  # Aumentado para ResNet
batch_size = 32
epochs = 20

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Transfer Learning com ResNet50
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Congelar camadas do modelo base
base_model.trainable = False

# Construção do modelo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(count_class(dataset_path), activation="softmax")
])

# Compilando o modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

# Treinamento
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Salvando o modelo treinado
model.save('model/my_model.keras')

# Resumo do treinamento
print(f"Treinamento finalizado com {history.history['accuracy'][-1] * 100:.2f}% de precisão.")
