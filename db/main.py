import os
from src import ImageProcessor, Files
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv(override=True)

    # Carregar variáveis de ambiente
    cx = os.getenv('ENGINE_ID')
    key = os.getenv('API_KEY')

    # Parâmetros de busca
    parametros = ['cat', 'dog']
    
    base_path = os.path.join('db', 'images')

    # Inicializar o processador de imagens
    #image_processor = ImageProcessor(cx, key, base_path, parametros, num_images=50)
    #image_processor.run()

    files = Files(base_path)
    files.run()
