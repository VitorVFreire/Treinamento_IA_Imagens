from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def categorical(path, imgW, imgH, batch_size, epochs, model_name):
    # Data augmentation
    trainImgs = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    testImgs = ImageDataGenerator(rescale=1.0 / 255)

    # Load datasets
    train = trainImgs.flow_from_directory(
        path + '/train', target_size=(imgW, imgH),
        batch_size=batch_size, class_mode='categorical', shuffle=True
    )

    test = testImgs.flow_from_directory(
        path + '/test', target_size=(imgW, imgH),
        batch_size=batch_size, class_mode='categorical', shuffle=False
    )

    classes = list(train.class_indices.keys())

    # Model definition
    model = Sequential()

    # First block
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(imgW, imgH, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Third block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Fourth block
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Global Average Pooling
    model.add(GlobalAveragePooling2D())

    # Fully connected layers
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(classes), activation='softmax'))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True)

    # Train model
    model.fit(
        train, 
        epochs=epochs, 
        validation_data=test, 
        validation_steps=test.samples // batch_size,
        steps_per_epoch=train.samples // batch_size,
        callbacks=[early_stopping, checkpoint]
    )

    # Save model
    model.save(model_name)