import os
from src import ImageProcessor, Files
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv(override=True)

    # Carregar variáveis de ambiente
    cx = os.getenv('ENGINE_ID')
    key = os.getenv('API_KEY')

    # Parâmetros de busca
    parametros = ['Beagle', 'Border collie', 'Boxer', 'Dálmata']
    aux_query = 'Dog '

    images_db = 'db/images_db.json'
    
    base_path = os.path.join('db', 'images')
    files = Files(base_path, ['train', 'test'], images_db)

    # Inicializar o processador de imagens
    image_processor = ImageProcessor(cx, key, base_path, parametros, aux_query, files.count_files_per_folder(),num_images=300)
    image_processor.run()

    files.run()
    files.write_json_file()