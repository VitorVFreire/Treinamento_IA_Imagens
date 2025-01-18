from .src import categorical
from db import Files
import time

start = time.time()

path = 'db/images'
imgW, imgH = 50, 50
batch_size = 64
epochs = 10
model_name = 'model/model.keras'
mode = 'categorical'

class_names = Files('db/images', ['train', 'test'])
class_names.load_names()

if mode == 'categorical':
    categorical(path, imgW, imgH, batch_size, epochs, model_name)
else:
    print('Nenhum Treinamento Realizado!!!')

end = time.time()
length = end - start

duration_time, unit = (length, 'segs') if length <= 100 else (length / 60, 'mins')

print(f'Tempo de Execução: {duration_time} {unit}!')