import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from db import Files
import time

def train_ai(target, train_dir, val_dir, batch_size, img_height, img_width):
    # Geradores de Dados com Augmentação
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Gerador de Dados de Treinamento
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',  # Usando classificação binária
    )

    # Gerador de Dados de Validação
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',  # Usando classificação binária
    )

    # Modelo de Rede Neural
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Saída binária com sigmoid
    ])

    # Compilação do Modelo
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Função de perda binária
        metrics=['accuracy']
    )

    # Callbacks para interrupção antecipada e checkpoint de modelo
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'model/{target}_binary_best.keras', save_best_only=True, monitor='val_loss', mode='min')

    # Treinamento do Modelo (sem workers e multiprocessamento diretamente)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=20,  # Número de épocas ajustado para 20
        callbacks=[early_stopping, checkpoint]
    )

    # Salvar o modelo final após o treinamento
    model.save(f'model/{target}_binary_final.keras')

    # Avaliar o modelo no conjunto de validação
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Validação - Perda: {val_loss}, Acurácia: {val_accuracy}")

start = time.time()

# Carregar os diretórios e nomes das classes
files = Files('db/images')
files.load_names()
classes_name = files.names

# Treinamento para cada classe
for class_name in classes_name:
    train_dir = f"db/images/train/{class_name}"
    val_dir = f"db/images/validation/{class_name}"

    # Parâmetros do treinamento
    img_height, img_width = 150, 150
    batch_size = 18  # Ajuste de batch size

    print(f'Treinando Modelo Binário para {class_name}...')
    train_ai(class_name, train_dir, val_dir, batch_size, img_height, img_width)

end = time.time()
length = end - start

print("Tempo de Execução: ", length, "segs!")