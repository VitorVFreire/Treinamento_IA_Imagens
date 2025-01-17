from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims, argmax
import time
from tensorflow.keras.optimizers import Adam

def train_ai(path, imgW, imgH, batch_size, epochs):
    trainImgs = ImageDataGenerator(
        rescale = 1./255, shear_range = 0.2, 
        zoom_range = 0.2, horizontal_flip = True)
    testImgs = ImageDataGenerator(rescale = 1./255)

    train = trainImgs.flow_from_directory(
            path + '/train', target_size = (imgW, imgH),
            batch_size = batch_size, class_mode = 'categorical')
    test = testImgs.flow_from_directory(
            path + '/test', target_size = (imgW, imgH),
            batch_size = batch_size, class_mode = 'categorical')

    classes = list(train.class_indices.keys())

    cnn = Sequential()

    cnn.add(Conv2D(
            32, (3, 3), input_shape = (imgW, imgH, 3),
            activation = 'relu'))  # Ajuste do kernel
    cnn.add(MaxPooling2D(
            pool_size = (2, 2)))  # Ajuste do downscale
    cnn.add(Conv2D(
            64, (3, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(
            pool_size = (2, 2)))  # Ajuste do downscale
    cnn.add(Flatten())

    cnn.add(Dense(
        units = 128, activation = 'relu'))
    cnn.add(Dense(
            units = len(classes), activation = 'softmax'))
    cnn.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])
    
    cnn.fit(
        train, epochs = epochs, 
        validation_data = test, validation_steps = 512)

    cnn.save('model/model.keras')

start = time.time()

imgW, imgH = 100, 100
path = 'db/images'

train_ai(path, imgW, imgH, 24, 10)

end = time.time()
length = end - start

print("Tempo de Execução: ", length, "segs!")
