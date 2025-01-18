import os
from src import ImageProcessor, Files
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv(override=True)

    # Carregar variáveis de ambiente
    cx = os.getenv('ENGINE_ID')
    key = os.getenv('API_KEY')

    # Parâmetros de busca
    parametros = ['Toyota Corolla', 'Hyundai HB20', 'Hyundai Creta', 'Fiat Uno', 'Chevrolet Corsa']
    aux_query = 'exterior view of '
    
    base_path = os.path.join('db', 'images')

    # Inicializar o processador de imagens
    image_processor = ImageProcessor(cx, key, base_path, parametros, aux_query, num_images=200)
    image_processor.run()

    files = Files(base_path, ['train', 'test'])
    files.run()