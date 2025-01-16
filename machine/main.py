import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def train_ai(target, train_dir, val_dir, batch_size, img_height, img_width):
    # Geradores de Dados
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'  # Aqui definimos como binário
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'  # Aqui também definimos como binário
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

    # Treinamento
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=10
    )

    # Salvar o modelo treinado
    model.save('model/my_model.keras')

    # Avaliar no conjunto de validação
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Validação - Perda: {val_loss}, Acurácia: {val_accuracy}")

# Caminhos para os diretórios
train_dir = "images/train"
val_dir = "images/validation"

# Parâmetros
img_height, img_width = 150, 150
batch_size = 32

train_ai('dog', train_dir, val_dir, batch_size, img_height, img_width)